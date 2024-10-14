use odht::{Config, FxHashFn, HashTable};

pub struct PrefixConfig;
pub struct EncoderConfig;
pub struct DecoderConfig;

const MAX_TOKEN_BYTES: usize = 128;

impl Config for PrefixConfig {
    type Key = i64;
    type Value = ();
    type EncodedKey = [u8; 8];
    type EncodedValue = [u8; 0];
    type H = FxHashFn;

    fn encode_key(k: &Self::Key) -> Self::EncodedKey { k.to_le_bytes() }
    fn encode_value(_: &Self::Value) -> Self::EncodedValue { [] }
    fn decode_key(k: &Self::EncodedKey) -> Self::Key { i64::from_le_bytes(*k) }
    fn decode_value(_: &Self::EncodedValue) -> Self::Value { () }
}

impl Config for EncoderConfig {
    type Key = Vec<u8>;
    type Value = usize;
    type EncodedKey = [u8; 8+MAX_TOKEN_BYTES];
    type EncodedValue = [u8; 8];
    type H = FxHashFn;

    fn encode_key(k: &Self::Key) -> Self::EncodedKey {
        let mut encoded_key = [0; 8 + MAX_TOKEN_BYTES];
        encoded_key[..8].copy_from_slice(&(k.len() as usize).to_le_bytes());
        encoded_key[8..8+k.len()].copy_from_slice(k);
        encoded_key
    }
    fn decode_key(k: &Self::EncodedKey) -> Self::Key {
        let len = usize::from_le_bytes(k[..8].try_into().unwrap());
        let mut key = k[8..].to_vec();
        key.resize(len, 0);
        key
    }

    fn encode_value(v: &Self::Value) -> Self::EncodedValue { v.to_le_bytes() }
    fn decode_value(v: &Self::EncodedValue) -> Self::Value { usize::from_le_bytes(*v) }
}

impl Config for DecoderConfig {
    type Key = usize;
    type Value = Vec<u8>;
    type EncodedKey = [u8; 8];
    type EncodedValue = [u8; 8+MAX_TOKEN_BYTES];
    type H = FxHashFn;

    fn encode_key(k: &Self::Key) -> Self::EncodedKey { k.to_le_bytes() }
    fn decode_key(k: &Self::EncodedKey) -> Self::Key { usize::from_le_bytes(*k) }

    fn encode_value(v: &Self::Value) -> Self::EncodedValue {
        let mut encoded_value = [0; 8+MAX_TOKEN_BYTES];
        encoded_value[..8].copy_from_slice(&(v.len() as usize).to_le_bytes());
        encoded_value[8..8+v.len()].copy_from_slice(v);
        encoded_value
    }
    fn decode_value(v: &Self::EncodedValue) -> Self::Value {
        let len = usize::from_le_bytes(v[..8].try_into().unwrap());
        let mut value = v[8..].to_vec();
        value.resize(len, 0);
        value
    }
}

pub trait HashTableExt<C: Config> {
    fn keys(&self) -> impl Iterator<Item = C::EncodedKey>;
    fn values(&self) -> impl Iterator<Item = C::EncodedValue>;
}

impl<C: Config> HashTableExt<C> for HashTable<C, &'static [u8]> {
    fn keys(&self) -> impl Iterator<Item = C::EncodedKey> {
        self.iter().map(|(k, _)| C::encode_key(&k))
    }

    fn values(&self) -> impl Iterator<Item = C::EncodedValue> {
        self.iter().map(|(_, v)| C::encode_value(&v))
    }
}