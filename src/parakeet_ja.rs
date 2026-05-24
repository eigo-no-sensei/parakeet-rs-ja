use crate::audio::{self, FeatureCache};
use crate::config::PreprocessorConfig;
use crate::decoder::{TimedToken, TranscriptionResult};
use crate::decoder_tdt_ja::{
    group_ja_by_sentences, group_ja_by_words, ParakeetJADecoder,
};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::ParakeetTDTModel;
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use crate::vocab::Vocabulary;
use std::path::{Path, PathBuf};

/// Japanese TDT model backed by `sunilmahendrakar/parakeet-tdt-0.6b-ja-onnx`
/// (or any compatible export of `nvidia/parakeet-tdt_ctc-0.6b-ja`).
///
/// # Expected directory layout
///
/// ```text
/// <model_dir>/
///   encoder-model.onnx          # FastConformer encoder  (aliases: encoder.onnx)
///   decoder_joint-model.onnx    # TDT decoder+joiner    (aliases: decoder_joint.onnx)
///   vocab.txt                   # SentencePiece tokens, one per line: "<token> <id>"
/// ```
///
/// # Usage
///
/// ```ignore
/// use parakeet_rs::{ParakeetJA, Transcriber, TimestampMode};
///
/// let mut model = ParakeetJA::from_pretrained("./ja-onnx", None)?;
///
/// let result = model.transcribe_file("speech.wav", Some(TimestampMode::Sentences))?;
/// println!("{}", result.text);
///
/// for seg in &result.tokens {
///     println!("[{:.2}s – {:.2}s] {}", seg.start, seg.end, seg.text);
/// }
/// ```
pub struct ParakeetJA {
    model: ParakeetTDTModel,
    decoder: ParakeetJADecoder,
    /// Kept separately so we can rebuild boundary-aware token lists for
    /// `Words`/`Sentences` grouping without repeating the full decode.
    vocab: Vocabulary,
    preprocessor_config: PreprocessorConfig,
    feature_cache: FeatureCache,
    model_dir: PathBuf,
}

impl ParakeetJA {
    /// Load the Japanese TDT model from `path`.
    ///
    /// `path` must be a directory containing `encoder-model.onnx` (or
    /// `encoder.onnx`), `decoder_joint-model.onnx` (or `decoder_joint.onnx`),
    /// and `vocab.txt`.
    ///
    /// Pass `None` for `config` to use the CPU execution provider with 4
    /// intra-op threads (same default as the English TDT model).
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(Error::Config(format!(
                "ParakeetJA: path must be a directory: {}",
                path.display()
            )));
        }

        let vocab_path = path.join("vocab.txt");
        if !vocab_path.exists() {
            return Err(Error::Config(format!(
                "ParakeetJA: vocab.txt not found in {}",
                path.display()
            )));
        }

        // The JA model uses the same FastConformer architecture as the English
        // TDT variant: 128 mel features, 16 kHz, 10 ms frame shift.
        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        let exec_config = config.unwrap_or_default();
        let vocab = Vocabulary::from_file(&vocab_path)?;
        let vocab_size = vocab.size();

        let model = ParakeetTDTModel::from_pretrained(path, exec_config, vocab_size)?;
        let decoder = ParakeetJADecoder::from_vocab(vocab.clone());
        let feature_cache = FeatureCache::from_config(&preprocessor_config);

        Ok(Self {
            model,
            decoder,
            vocab,
            preprocessor_config,
            feature_cache,
            model_dir: path.to_path_buf(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn preprocessor_config(&self) -> &PreprocessorConfig {
        &self.preprocessor_config
    }

    /// Rebuild a boundary-aware (▁-prefixed) token list from raw decode output.
    ///
    /// The regular `decode_with_timestamps` strips `▁` immediately.  For word
    /// and sentence grouping we need the marker intact so `group_ja_by_words`
    /// and `group_ja_by_sentences` can detect morpheme boundaries.
    fn boundary_tokens(
        &self,
        token_ids: &[usize],
        frame_indices: &[usize],
    ) -> Vec<TimedToken> {
        let hop = self.preprocessor_config.hop_length;
        let sr = self.preprocessor_config.sampling_rate;
        const ENCODER_STRIDE: usize = 8;

        token_ids
            .iter()
            .zip(frame_indices.iter())
            .enumerate()
            .filter_map(|(i, (&tok_id, &frame))| {
                let text = self.vocab.id_to_text(tok_id)?;
                // Drop structural special tokens (keep <unk>).
                if text.starts_with('<') && text.ends_with('>') && text != "<unk>" {
                    return None;
                }
                let start = (frame * ENCODER_STRIDE * hop) as f32 / sr as f32;
                let end = frame_indices
                    .get(i + 1)
                    .map(|&nf| (nf * ENCODER_STRIDE * hop) as f32 / sr as f32)
                    .unwrap_or(start + 0.01);
                Some(TimedToken {
                    text: text.to_string(), // ▁ intact
                    start,
                    end,
                })
            })
            .collect()
    }
}

impl Transcriber for ParakeetJA {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult> {
        let features = audio::extract_features_with_cache(
            audio,
            sample_rate,
            channels,
            &self.preprocessor_config,
            &self.feature_cache,
        )?;

        let (token_ids, frame_indices, durations) = self.model.forward(features)?;
        let mode = mode.unwrap_or(TimestampMode::Tokens);

        match mode {
            TimestampMode::Tokens => {
                // Fast path: decode directly, strip ▁, return.
                self.decoder.decode_with_timestamps(
                    &token_ids,
                    &frame_indices,
                    &durations,
                    self.preprocessor_config.hop_length,
                    self.preprocessor_config.sampling_rate,
                )
            }

            TimestampMode::Words => {
                let raw = self.boundary_tokens(&token_ids, &frame_indices);
                let grouped = group_ja_by_words(&raw);
                let text: String = grouped.iter().map(|t| t.text.as_str()).collect();
                Ok(TranscriptionResult {
                    text,
                    tokens: grouped,
                })
            }

            TimestampMode::Sentences => {
                let raw = self.boundary_tokens(&token_ids, &frame_indices);
                let grouped = group_ja_by_sentences(&raw);
                let text: String = grouped.iter().map(|t| t.text.as_str()).collect();
                Ok(TranscriptionResult {
                    text,
                    tokens: grouped,
                })
            }
        }
    }
}
