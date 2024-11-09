use crate::corebpe::CoreBPE;
use regex::Regex;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use thiserror::Error;
use odht::HashTableOwned;
use crate::rollhash::{roll_hash, roll_hash_slice};
use crate::corebpe::Rank;
use std::sync::Arc;

include!("odht.rs");
include!(concat!(env!("OUT_DIR"), "/odht_gen.rs"));

/// A struct that represents an encoding scheme based on byte-pair encoding (BPE).
#[derive(Debug)]
pub struct Encoding {
    /// The name of the encoding.
    pub name: String,
    /// The maximum length of the keys in `mergeable_ranks`.
    mergeable_ranks_max_key_len: usize,
    /// All prefixes of the mergeable ranks. May or may not be tokens themselves!
    prefixes_of_mergeable_ranks: HashTableOwned<PrefixConfig>,
    /// The map from special token strings to their values.
    special_tokens: HashMap<String, Rank>,
    /// The core BPE logic implemented in Rust.
    core_bpe: Arc<CoreBPE>,
}

// TODO: make a non-generic encoding error here
#[derive(Error, Debug, Clone)]
pub enum EncodingError {
    #[error("encoding: {0}")]
    GenericEncodingError(String),
}

#[derive(Debug, Clone)]
pub enum SpecialTokenAction {
    /// The special token is forbidden. If it is included in the string, an error will be returned.
    Forbidden,
    /// The special token is tokenized as normal text.
    NormalText,
    /// The special token is treated as the special token it is. If this is applied to a specific text and the text is NOT a special token then an error will be returned. If it is the default action no error will be returned, don't worry.
    Special,
}

#[derive(Debug, Clone)]
pub struct SpecialTokenHandling {
    pub default: SpecialTokenAction,
    pub overrides: Vec<(String, SpecialTokenAction)>,
}

impl Default for SpecialTokenHandling {
    fn default() -> Self {
        Self {
            default: SpecialTokenAction::Forbidden,
            overrides: vec![],
        }
    }
}

impl Encoding {
    /// Creates a new encoding from the given parameters.
    pub fn new(
        name: &str,
        pat_str: &str,
        mergeable_ranks: HashMap<Vec<u8>, Rank>,
        special_tokens: HashMap<String, Rank>,
        explicit_n_vocab: Option<usize>,
    ) -> Result<Self, EncodingError> {
        let max_token_value = match mergeable_ranks
            .values()
            .chain(special_tokens.values())
            .max()
            .copied()
        {
            Some(value) => value,
            None => return Err(EncodingError::GenericEncodingError("No token values found".to_string())),
        };
        if let Some(explicit_n_vocab) = explicit_n_vocab {
            if mergeable_ranks.len() + special_tokens.len() != explicit_n_vocab {
                return Err(EncodingError::GenericEncodingError("Mismatch between explicit vocab size and actual vocab size".to_string()));
            }
            if max_token_value as usize != explicit_n_vocab - 1 {
                return Err(EncodingError::GenericEncodingError("Mismatch between max token value and explicit vocab size".to_string()));
            }
        }

        let mergeable_ranks_max_key_len = mergeable_ranks
            .keys()
            .map(|bytes| bytes.len())
            .max()
            .ok_or_else(|| EncodingError::GenericEncodingError("No mergeable ranks found".to_string()))?;

        let core_bpe = CoreBPE::new(
            mergeable_ranks,
            special_tokens.clone(),
            pat_str,
        )
        .map_err(|e| EncodingError::GenericEncodingError(format!("Error creating core BPE: {}", e)))?;

        let prefixes_of_mergeable_ranks = unsafe {
            HashTableOwned::<PrefixConfig>::from_raw_bytes_unchecked(match name {
                "r50k_base" => data::R50K_BASE_PREFIXES_ODHT,
                "p50k_base" => data::P50K_BASE_PREFIXES_ODHT,
                "cl100k_base" => data::CL100K_BASE_PREFIXES_ODHT,
                "o200k_base" => data::O200K_BASE_PREFIXES_ODHT,
                "codestral" => data::CODESTRAL_PREFIXES_ODHT,
                "llama3" => data::LLAMA3_PREFIXES_ODHT,
                "deepseekv2" => data::DEEPSEEKV2_PREFIXES_ODHT,
                _ => return Err(EncodingError::GenericEncodingError(format!("Embedded prefix table not found for encoding: {}", name))),
            })
        };

        Ok(Self {
            name: name.to_string(),
            mergeable_ranks_max_key_len,
            prefixes_of_mergeable_ranks,
            special_tokens,
            core_bpe: Arc::new(core_bpe),
        })
    }

    /// Encodes a string into tokens, ignoring special tokens.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        self.core_bpe.encode_ordinary(text)
    }

    pub fn estimate_num_tokens_no_special_tokens_fast(&self, text: &str, replace_spaces_with_lower_one_eighth_block: bool) -> usize {
        let tokens = if replace_spaces_with_lower_one_eighth_block {
            self.count_tokens(&text.replace(" ", "\u{2581}"))
        } else {
            self.count_tokens(text)
        };

        tokens
    }

    fn count_tokens(&self, text: &str) -> usize {
        let mut token_count = 0;
        let mut current_token = Vec::new();
        let mut current_token_hash: i64 = 0;
        let mut new_current_token = Vec::new();

        for byte in text.bytes() {
            current_token.push(byte);
            current_token_hash = roll_hash(current_token_hash, byte);

            // if the current token is longer than the maximum mergeable rank key length
            // or if the current token is not in the prefixes of mergeable ranks,
            // we need to split the current token and begin actually checking for the largest
            // mergeable prefix
            while !self.prefixes_of_mergeable_ranks.contains_key(&current_token_hash)
                || current_token.len() > self.mergeable_ranks_max_key_len
            {
                if current_token.len() > 1 {
                    new_current_token.clear();
                    new_current_token.push(current_token.pop().unwrap());
                    while !self.core_bpe.encoder.contains_key(&current_token) {
                        if current_token.len() == 1 {
                            break;
                        }
                        new_current_token.push(current_token.pop().unwrap());
                    }
                    current_token.clear();
                    // reverse new_current_token
                    new_current_token.reverse();
                    // swap new_current_token and current_token
                    std::mem::swap(&mut new_current_token, &mut current_token);
                    current_token_hash = roll_hash_slice(&current_token);
                } else {
                    current_token.clear();
                    current_token_hash = 0;
                }
                token_count += 1;
            }
        }

        while !self.core_bpe.encoder.contains_key(&current_token) {
            if current_token.len() == 0 {
                break;
            }
            if current_token.len() > 1 {
                new_current_token.clear();
                new_current_token.push(current_token.pop().unwrap());
                while !self.core_bpe.encoder.contains_key(&current_token) {
                    if current_token.len() == 1 {
                        break;
                    }
                    new_current_token.push(current_token.pop().unwrap());
                }

                current_token.clear();
                // reverse new_current_token
                new_current_token.reverse();
                // swap new_current_token and current_token
                std::mem::swap(&mut new_current_token, &mut current_token);
            } else {
                current_token.clear();
            }
            token_count += 1;
        }

        if current_token.len() > 0 {
            token_count += 1;
        }

        token_count
    }

    /// Encodes a string into tokens.
    ///
    /// Special tokens are artificial tokens used to unlock capabilities from a model,
    /// such as fill-in-the-middle. So we want to be careful about accidentally encoding special
    /// tokens, since they can be used to trick a model into doing something we don't want it to do.
    pub fn encode(
        &self,
        text: &str,
        special_token_handling: &SpecialTokenHandling,
    ) -> Result<Vec<Rank>, EncodingError> {
        // first check if all special tokens are valid
        for (special_token, _) in &special_token_handling.overrides {
            if !self.special_tokens.contains_key(special_token) {
                return Err(EncodingError::GenericEncodingError(format!(
                    "Unknown special token {:?}",
                    special_token
                )));
            }
        }

        let special_tokens_to_recognize = match special_token_handling.default {
            SpecialTokenAction::Special => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Special => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Special => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        let forbidden_special = match special_token_handling.default {
            SpecialTokenAction::Forbidden => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Forbidden => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Forbidden => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        if !forbidden_special.is_empty() {
            let re = special_token_regex(&forbidden_special);
            if let Some(matched) = re.find(text) {
                return Err(EncodingError::GenericEncodingError(format!(
                    "Encountered text corresponding to disallowed special token {:?}.",
                    matched.as_str()
                )));
            }
        }

        Ok(self.core_bpe.encode(text, &special_tokens_to_recognize))
    }

    /// Encodes a list of strings into tokens, in parallel, ignoring special tokens.
    pub fn encode_ordinary_batch(&self, _text: Vec<String>) -> Vec<Vec<usize>> {
        // encoder = functools.partial(self.encode_ordinary)
        // with ThreadPoolExecutor(num_threads) as e:
        //     return list(e.map(encoder, text))
        // TODO: use rayon
        unimplemented!("todo")
    }

    /// Encodes a list of strings into tokens, in parallel.
    pub fn encode_batch(
        &self,
        _text: Vec<String>,
        _special_token_handling: &SpecialTokenHandling,
    ) -> Vec<Vec<usize>> {
        // with ThreadPoolExecutor(num_threads) as e:
        //     return list(e.map(encoder, text))

        // TODO: use rayon
        unimplemented!("todo")
    }

    /// Encodes a string into stable tokens and possible completion sequences.
    ///
    /// Note that the stable tokens will only represent a substring of `text`.
    ///
    /// This API should itself be considered unstable.
    pub fn encode_with_unstable(
        &self,
        text: &str,
        special_token_handling: &SpecialTokenHandling,
    ) -> Result<(Vec<Rank>, HashSet<Vec<Rank>>), EncodingError> {
        // first check if all special tokens are valid
        for (special_token, _) in &special_token_handling.overrides {
            if !self.special_tokens.contains_key(special_token) {
                return Err(EncodingError::GenericEncodingError(format!(
                    "Unknown special token {:?}",
                    special_token
                )));
            }
        }

        let special_tokens_to_recognize = match special_token_handling.default {
            SpecialTokenAction::Special => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Special => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Special => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        let forbidden_special = match special_token_handling.default {
            SpecialTokenAction::Forbidden => {
                &self
                    .special_tokens
                    .keys()
                    .map(|s| s.as_str())
                    .collect::<HashSet<_>>()
                    - &special_token_handling
                        .overrides
                        .iter()
                        .filter_map(|(token, action)| match action {
                            SpecialTokenAction::Forbidden => None,
                            _ => Some(token.as_str()),
                        })
                        .collect::<HashSet<_>>()
            }
            _ => special_token_handling
                .overrides
                .iter()
                .filter_map(|(token, action)| match action {
                    SpecialTokenAction::Forbidden => Some(token.as_str()),
                    _ => None,
                })
                .collect::<HashSet<_>>(),
        };
        if !forbidden_special.is_empty() {
            let re = special_token_regex(&forbidden_special);
            if let Some(matched) = re.find(text) {
                return Err(EncodingError::GenericEncodingError(format!(
                    "Encountered text corresponding to disallowed special token {:?}.",
                    matched.as_str()
                )));
            }
        }

        Ok(self
            .core_bpe
            .encode_with_unstable(text, &special_tokens_to_recognize))
    }

    /// Encodes text corresponding to a single token to its token value.
    ///
    /// NOTE: this will encode all special tokens.
    ///
    /// Returns an error if the token is not in the vocabulary.
    pub fn encode_single_token(&self, text: &str) -> Result<Rank, Vec<u8>> {
        self.encode_single_token_bytes(text.as_bytes())
    }
    pub fn encode_single_token_bytes(&self, bytes: &[u8]) -> Result<Rank, Vec<u8>> {
        self.core_bpe.encode_single_token(bytes)
    }

    /// Decodes a list of tokens into bytes.
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Vec<u8> {
        self.core_bpe.decode_bytes(tokens)
    }

    /// Decodes a list of tokens into a string.
    ///
    /// WARNING: the default behaviour of this function is lossy, since decoded bytes are not
    /// guaranteed to be valid UTF-8.
    pub fn decode(&self, tokens: &[Rank]) -> String {
        let bytes = self.core_bpe.decode_bytes(tokens);
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Decodes a token into bytes.
    ///
    /// NOTE: this will decode all special tokens.
    ///
    /// Returns an error if the token is not in the vocabulary.
    pub fn decode_single_token_bytes(&self, token: Rank) -> Result<Vec<u8>, Rank> {
        self.core_bpe.decode_single_token_bytes(token)
    }

    /// Decodes a list of tokens into a list of bytes.
    ///
    /// Useful for visualising tokenisation.
    ///
    /// Returns an error if any of the tokens is not in the vocabulary.
    pub fn decode_tokens_bytes(&self, tokens: Vec<Rank>) -> Result<Vec<Vec<u8>>, Rank> {
        tokens
            .into_iter()
            .map(|token| self.decode_single_token_bytes(token))
            .collect()
    }

    /// Returns the list of all token byte values.
    pub fn token_byte_values(&self) -> Vec<Vec<u8>> {
        self.core_bpe.token_byte_values()
    }

    /// Returns the end-of-text token.
    pub fn eot_token(&self) -> Rank {
        self.special_tokens["<|endoftext|>"]
    }

    /// Encodes text corresponding to bytes without a regex split.
    ///
    /// NOTE: this will not encode any special tokens.
    pub fn _encode_single_piece(&self, text: &str) -> Vec<Rank> {
        let text_or_bytes = text.as_bytes();
        self.core_bpe.encode_single_piece(text_or_bytes)
    }

    /// Encodes bytes into tokens.
    fn _encode_bytes(&self, text: &[u8]) -> Vec<Rank> {
        self.core_bpe._encode_bytes(text)
    }
}

/// Returns a regular expression that matches any of the given special tokens.
fn special_token_regex(tokens: &HashSet<&str>) -> Regex {
    let inner = tokens
        .iter()
        .map(|token| regex::escape(token))
        .collect::<Vec<_>>()
        .join("|");
    Regex::new(&format!("({})", inner)).unwrap()
}

// only use for testing!!!!
impl Default for Encoding {
    fn default() -> Self {
        crate::openai_public::EncodingFactory::cl100k_base().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{EncodingFactory, EncodingFactoryError};

    use super::*;
    use test_case::test_case;
    use memory_stats::memory_stats;

    #[test_case(EncodingFactory::llama3 ; "llama3")]
    #[test_case(EncodingFactory::codestral ; "codestral")]
    #[test_case(EncodingFactory::cl100k_im ; "cl100k_im")]
    #[test_case(EncodingFactory::cl100k_base ; "cl100k_base")]
    fn test_encoding_memory_usage(encoding_factory: fn() -> Result<Encoding, EncodingFactoryError>) {
        let initial_memory = memory_stats().unwrap().physical_mem;

        let _ = encoding_factory().unwrap();

        let final_memory = memory_stats().unwrap().physical_mem;
        let memory_used = final_memory - initial_memory;

        println!("Memory used by {}: {} bytes", std::any::type_name_of_val(&encoding_factory), memory_used);

        // You can set a threshold based on your requirements
        assert!(memory_used < 600 * 1024 * 1024, "Memory usage exceeded 600MB");
    }
}
