use odht::{Config, FxHashFn};

pub struct PrefixConfig;

impl Config for PrefixConfig {
    type Key = i64;
    type Value = ();
    type EncodedKey = [u8; 8];
    type EncodedValue = [u8; 0];
    type H = FxHashFn;

    #[inline(always)]
    fn encode_key(k: &Self::Key) -> Self::EncodedKey { k.to_le_bytes() }
    #[inline(always)]
    fn encode_value(_: &Self::Value) -> Self::EncodedValue { [] }
    #[inline(always)]
    fn decode_key(k: &Self::EncodedKey) -> Self::Key { i64::from_le_bytes(*k) }
    #[inline(always)]
    fn decode_value(_: &Self::EncodedValue) -> Self::Value { () }
}