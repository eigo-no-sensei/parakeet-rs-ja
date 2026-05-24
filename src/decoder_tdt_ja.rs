use crate::decoder::{TimedToken, TranscriptionResult};
use crate::error::Result;
use crate::vocab::Vocabulary;

/// TDT decoder for Japanese Parakeet models.
///
/// Differs from the English `ParakeetTDTDecoder` in three ways:
///
/// 1. SentencePiece `▁` boundary markers are **stripped** (not replaced with a
///    space) because Japanese orthography has no inter-word spaces.
/// 2. `TimestampMode::Words` groups tokens by `▁`-prefixed morpheme boundaries
///    rather than ASCII space separators.
/// 3. `TimestampMode::Sentences` uses Japanese sentence-ending punctuation
///    (`。`, `！`, `？`) in addition to ASCII `!` and `?`.
#[derive(Debug)]
pub struct ParakeetJADecoder {
    vocab: Vocabulary,
}

impl ParakeetJADecoder {
    pub fn from_vocab(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    /// Expose the raw vocab string (including any `▁` prefix) for `id`.
    ///
    /// Used by `parakeet_ja.rs` when rebuilding boundary-aware token lists
    /// for word/sentence grouping without re-decoding from scratch.
    pub(crate) fn vocab_text(&self, id: usize) -> Option<&str> {
        self.vocab.id_to_text(id)
    }

    /// Decode token IDs + frame timing into a `TranscriptionResult`.
    ///
    /// Token texts are stored with `▁` stripped but **no space inserted**,
    /// ready for direct Japanese text concatenation.
    pub fn decode_with_timestamps(
        &self,
        tokens: &[usize],
        frame_indices: &[usize],
        _durations: &[usize],
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        // TDT encoder applies 8× temporal subsampling before the decoder.
        const ENCODER_STRIDE: usize = 8;

        let mut timed_tokens: Vec<TimedToken> = Vec::new();
        let mut full_text = String::new();

        for (i, &token_id) in tokens.iter().enumerate() {
            let token_text = match self.vocab.id_to_text(token_id) {
                Some(t) => t,
                None => continue,
            };

            // Skip structural special tokens (e.g. <pad>, <s>, </s>) but keep
            // <unk> so transcription failures are visible.
            if token_text.starts_with('<')
                && token_text.ends_with('>')
                && token_text != "<unk>"
            {
                continue;
            }

            let frame = frame_indices[i];
            let start =
                (frame * ENCODER_STRIDE * hop_length) as f32 / sample_rate as f32;
            let end = if i + 1 < frame_indices.len() {
                (frame_indices[i + 1] * ENCODER_STRIDE * hop_length) as f32
                    / sample_rate as f32
            } else {
                start + 0.01
            };

            // Strip SentencePiece word-boundary marker.  For Japanese we do
            // not insert a space — the boundary is implicit in the script.
            let display_text = token_text.replace('▁', "");

            full_text.push_str(&display_text);

            timed_tokens.push(TimedToken {
                text: display_text,
                start,
                end,
            });
        }

        Ok(TranscriptionResult {
            text: full_text,
            tokens: timed_tokens,
        })
    }
}

// ── Timestamp grouping ────────────────────────────────────────────────────────

/// Group raw token timestamps into morpheme-boundary segments for Japanese.
///
/// Expects `raw_tokens` to still carry the `▁` prefix (i.e. the token list
/// reconstructed directly from the vocabulary before stripping).  Tokens that
/// start with `▁` trigger a group boundary; continuation tokens are appended
/// to the current group.  The emitted `TimedToken::text` values have `▁`
/// stripped.
pub fn group_ja_by_words(raw_tokens: &[TimedToken]) -> Vec<TimedToken> {
    if raw_tokens.is_empty() {
        return Vec::new();
    }

    let mut words: Vec<TimedToken> = Vec::new();
    let mut current_text = String::new();
    let mut current_start = raw_tokens[0].start;

    for (i, tok) in raw_tokens.iter().enumerate() {
        let has_boundary = tok.text.starts_with('▁') || tok.text.starts_with(' ');

        if has_boundary && !current_text.is_empty() {
            words.push(TimedToken {
                text: strip_boundary_marker(&current_text),
                start: current_start,
                end: raw_tokens[i - 1].end,
            });
            current_text.clear();
            current_start = tok.start;
        } else if i == 0 {
            current_start = tok.start;
        }

        current_text.push_str(&tok.text);
    }

    if !current_text.is_empty() {
        words.push(TimedToken {
            text: strip_boundary_marker(&current_text),
            start: current_start,
            end: raw_tokens.last().unwrap().end,
        });
    }

    words
}

/// Group Japanese morpheme-boundary tokens into sentences.
///
/// Sentence boundaries: `。` `！` `？` `…` and ASCII `!` `?`.
/// A trailing fragment with no terminator is emitted as a final sentence.
pub fn group_ja_by_sentences(raw_tokens: &[TimedToken]) -> Vec<TimedToken> {
    let words = group_ja_by_words(raw_tokens);
    if words.is_empty() {
        return Vec::new();
    }

    let mut sentences: Vec<TimedToken> = Vec::new();
    let mut buf = String::new();
    let mut seg_start = words[0].start;
    let mut seg_end = words[0].end;

    for word in &words {
        buf.push_str(&word.text);
        seg_end = word.end;

        if ends_japanese_sentence(&word.text) {
            sentences.push(TimedToken {
                text: buf.clone(),
                start: seg_start,
                end: seg_end,
            });
            buf.clear();
            seg_start = seg_end;
        }
    }

    if !buf.is_empty() {
        sentences.push(TimedToken {
            text: buf,
            start: seg_start,
            end: seg_end,
        });
    }

    sentences
}

fn strip_boundary_marker(s: &str) -> String {
    s.replace('▁', "").replace(' ', "")
}

/// Returns `true` if `text` ends with a Japanese or ASCII sentence terminator.
fn ends_japanese_sentence(text: &str) -> bool {
    matches!(
        text.chars().last(),
        Some('。' | '！' | '？' | '…' | '.' | '!' | '?')
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(tokens: &[&str]) -> Vocabulary {
        Vocabulary {
            id_to_token: tokens.iter().map(|s| s.to_string()).collect(),
            _blank_id: 0,
        }
    }

    fn tok(text: &str, start: f32, end: f32) -> TimedToken {
        TimedToken {
            text: text.to_string(),
            start,
            end,
        }
    }

    // ── decoder ──────────────────────────────────────────────────────────────

    #[test]
    fn no_spaces_inserted_between_tokens() {
        let vocab = make_vocab(&["▁これ", "は", "▁テスト", "です", "。", "<blk>"]);
        let decoder = ParakeetJADecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                &[1, 1, 1, 1, 1],
                160,
                16000,
            )
            .unwrap();
        assert_eq!(result.text, "これはテストです。");
    }

    #[test]
    fn special_tokens_skipped() {
        let vocab = make_vocab(&["<pad>", "▁こんにちは", "<s>", "世界", "</s>", "<blk>"]);
        let decoder = ParakeetJADecoder::from_vocab(vocab);
        let result = decoder
            .decode_with_timestamps(
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3, 4],
                &[1, 1, 1, 1, 1],
                160,
                16000,
            )
            .unwrap();
        assert_eq!(result.text, "こんにちは世界");
    }

    // ── word grouping ────────────────────────────────────────────────────────

    #[test]
    fn group_by_words_splits_on_boundary() {
        // raw_tokens retain ▁ prefix
        let tokens = vec![
            tok("▁これ", 0.0, 0.1),
            tok("は", 0.1, 0.2),
            tok("▁テスト", 0.2, 0.4),
            tok("です", 0.4, 0.6),
            tok("。", 0.6, 0.65),
        ];
        let words = group_ja_by_words(&tokens);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "これは");
        assert_eq!(words[1].text, "テストです");
        assert_eq!(words[2].text, "。");
    }

    // ── sentence grouping ────────────────────────────────────────────────────

    #[test]
    fn group_by_sentences_splits_on_kuten() {
        let tokens = vec![
            tok("▁これ", 0.0, 0.1),
            tok("は", 0.1, 0.2),
            tok("。", 0.2, 0.25),
            tok("▁次", 0.25, 0.4),
            tok("の", 0.4, 0.5),
            tok("▁文", 0.5, 0.6),
            tok("。", 0.6, 0.65),
        ];
        let sentences = group_ja_by_sentences(&tokens);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "これは。");
        assert_eq!(sentences[1].text, "次の文。");
    }

    #[test]
    fn trailing_fragment_emitted() {
        let tokens = vec![tok("▁未完了", 0.0, 0.5), tok("の", 0.5, 0.6)];
        let sentences = group_ja_by_sentences(&tokens);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "未完了の");
    }
}
