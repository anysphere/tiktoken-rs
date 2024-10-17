use crate::encoding::Encoding;
use rustc_hash::FxHashMap as HashMap;
use thiserror::Error;
use crate::corebpe::Rank;

include!(concat!(env!("OUT_DIR"), "/encoders_gen.rs"));

#[derive(Error, Debug, Clone)]
pub enum EncodingFactoryError {
  #[error("failed to load encoding")]
  FailedToLoadEncoding,
  #[error("unable to create encoding: {0}")]
  UnableToCreateEncoding(String),
}

const ENDOFTEXT: &str = "<|endoftext|>";
const FIM_PREFIX: &str = "<|fim_prefix|>";
const FIM_MIDDLE: &str = "<|fim_middle|>";
const FIM_SUFFIX: &str = "<|fim_suffix|>";
const ENDOFPROMPT: &str = "<|endofprompt|>";

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const IM_SEP: &str = "<|im_sep|>";

#[derive(Clone, Debug, Copy)]
pub struct EncodingFactory {}
impl EncodingFactory {
  // The pattern in the original GPT-2 release is:
  // r"'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  // This is equivalent, but executes faster:
  const LEGACY_SPLITTER_REGEX: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

  pub fn gpt2() -> Result<Encoding, EncodingFactoryError> {
    // todo!
    // vocab_bpe_file: sha256 = 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
    // encoder_json_file: sha256 = 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
    // Encoding::new()
    unimplemented!("gpt2")
  }

  pub fn r50k_base() -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::R50K_BASE.iter().copied());
    let mut special_tokens: HashMap<String, Rank> =
      [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect();
    special_tokens.shrink_to_fit();
    Encoding::new(
      "r50k_base",
      EncodingFactory::LEGACY_SPLITTER_REGEX,
      mergeable_ranks,
      special_tokens,
      Some(50257),
    )
    .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn p50k_base() -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::P50K_BASE.iter().copied());
    let mut special_tokens: HashMap<String, Rank> =
      [(ENDOFTEXT.to_string(), 50256)].iter().cloned().collect();
    special_tokens.shrink_to_fit();
    Encoding::new(
      "p50k_base",
      EncodingFactory::LEGACY_SPLITTER_REGEX,
      mergeable_ranks,
      special_tokens,
      Some(50281),
    )
    .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  // we're just mirroring the official tiktoken. but i think this is slightly wrong for the latest models. in particular the end of text token appears to not be translated by the production tokenizer anymore
  pub fn cl100k_base() -> Result<Encoding, EncodingFactoryError> {
    EncodingFactory::cl100k_with_special_tokens(&[
      (ENDOFTEXT.to_string(), 100257),
      (FIM_PREFIX.to_string(), 100258),
      (FIM_MIDDLE.to_string(), 100259),
      (FIM_SUFFIX.to_string(), 100260),
      (ENDOFPROMPT.to_string(), 100276),
    ])
  }

  pub fn cl100k_im() -> Result<Encoding, EncodingFactoryError> {
    EncodingFactory::cl100k_with_special_tokens(&[
      // end of text actually doesn't appear to be supported by the latest models! you can try by sending it in in the completion model and counting tokens
      // (ENDOFTEXT.to_string(), 100257),
      (FIM_PREFIX.to_string(), 100258),
      (FIM_MIDDLE.to_string(), 100259),
      (FIM_SUFFIX.to_string(), 100260),
      (IM_START.to_string(), 100264),
      (IM_END.to_string(), 100265),
      (IM_SEP.to_string(), 100266),
      (ENDOFPROMPT.to_string(), 100276),
    ])
  }

  pub fn cl100k_with_special_tokens(
    special_tokens: &[(String, Rank)],
  ) -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::CL100K_BASE.iter().copied());
    let mut special_tokens: HashMap<String, Rank> = special_tokens.iter().cloned().collect();
    special_tokens.shrink_to_fit();
    // use faster version from tiktoken upstream https://github.com/openai/tiktoken/pull/258/files#r1487668172
    // const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
    Encoding::new(
            "cl100k_base",
            PATTERN,
            mergeable_ranks,
            special_tokens,
            None,
        )
        .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn o200k_with_special_tokens(
    special_tokens: &[(String, Rank)],
  ) -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::O200K_BASE.iter().copied());
    let mut special_tokens: HashMap<String, Rank> = special_tokens.iter().cloned().collect();
    special_tokens.shrink_to_fit();

    let pat_str: &str = &[
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ].join("|");

    Encoding::new("o200k_base", pat_str, mergeable_ranks, special_tokens, None)
      .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn codestral() -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::CODESTRAL.iter().copied());
    let special_tokens: HashMap<String, Rank> = [].iter().cloned().collect();

    Encoding::new("codestral", r"", mergeable_ranks, special_tokens, None)
      .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn deepseekv2() -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::DEEPSEEKV2.iter().copied());
    let special_tokens: HashMap<String, Rank> = [].iter().cloned().collect();

    Encoding::new("deepseekv2", r"", mergeable_ranks, special_tokens, None)
      .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn llama3() -> Result<Encoding, EncodingFactoryError> {
    let mergeable_ranks = HashMap::from_iter(data::LLAMA3.iter().copied());

    let num_base_tokens = mergeable_ranks.len();
    let mut special_tokens = vec![
      "<|begin_of_text|>".to_string(),
      "<|end_of_text|>".to_string(),
      "<|reserved_special_token_0|>".to_string(),
      "<|reserved_special_token_1|>".to_string(),
      "<|reserved_special_token_2|>".to_string(),
      "<|reserved_special_token_3|>".to_string(),
      "<|start_header_id|>".to_string(),
      "<|end_header_id|>".to_string(),
      "<|reserved_special_token_4|>".to_string(),
      "<|eot_id|>".to_string(), // end of turn
    ];

    let num_reserved_special_tokens = 256;
    special_tokens.extend(
      (5..num_reserved_special_tokens - 5).map(|i| format!("<|reserved_special_token_{}|>", i as Rank)),
    );

    let mut special_tokens_map: HashMap<String, Rank> =
      special_tokens.into_iter().enumerate().map(|(i, token)| (token, (num_base_tokens + i) as Rank)).collect();
    special_tokens_map.shrink_to_fit();

    let pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    let vocab_size = num_base_tokens + special_tokens_map.len();
    Encoding::new(
      "llama3",
      pat_str,
      mergeable_ranks,
      special_tokens_map,
      Some(vocab_size),
    )
    .map_err(|e| EncodingFactoryError::UnableToCreateEncoding(e.to_string()))
  }

  pub fn o200k_base() -> Result<Encoding, EncodingFactoryError> {
    EncodingFactory::o200k_with_special_tokens(&[
      (ENDOFTEXT.to_string(), 199999),
      (FIM_PREFIX.to_string(), 200000),
      (FIM_MIDDLE.to_string(), 200001),
      (FIM_SUFFIX.to_string(), 200002),
      (ENDOFPROMPT.to_string(), 200018),
    ])
  }

  pub fn o200k_im() -> Result<Encoding, EncodingFactoryError> {
    EncodingFactory::o200k_with_special_tokens(&[
      (ENDOFTEXT.to_string(), 199999),
      (FIM_PREFIX.to_string(), 200000),
      (FIM_MIDDLE.to_string(), 200001),
      (FIM_SUFFIX.to_string(), 200002),
      (IM_START.to_string(), 200006),
      (IM_END.to_string(), 200007),
      (IM_SEP.to_string(), 200008),
      (ENDOFPROMPT.to_string(), 200018),
    ])
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_encoding_encode_decode() {
    let encoding = EncodingFactory::cl100k_im().unwrap();
    let text = "Hello, world!";
    let tokens = encoding.encode_ordinary(text);
    let decoded = encoding.decode(&tokens);
    assert_eq!(text, decoded);
  }

  #[test]
  fn test_encoding_special_tokens() {
    let encoding = EncodingFactory::cl100k_im().unwrap();
    let special_tokens = vec!["<|im_start|>", "<|im_end|>", "<|im_sep|>", "<|endofprompt|>"];

    for token in special_tokens {
      let encoded = encoding.encode_single_token(token).unwrap();
      let decoded = encoding.decode_single_token_bytes(encoded).unwrap();
      assert_eq!(token.as_bytes(), decoded.as_slice());
    }
  }
}
