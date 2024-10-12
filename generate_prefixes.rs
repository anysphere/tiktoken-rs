use std::fs::File;
use std::io::Write;
use std::path::Path;
use tiktoken::{Encoding, EncodingFactory};
use odht::{HashTableOwned, Config, FxHashFn};

struct PrefixConfig;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encodings = [
        ("cl100k_base", EncodingFactory::cl100k_base()?),
        ("llama3", EncodingFactory::llama3()?),
        ("o200k_base", EncodingFactory::o200k_base()?),
        ("codestral", EncodingFactory::codestral()?),
    ];

    for (name, encoding) in encodings.iter() {
        generate_encoding_data(name, encoding)?;
    }

    println!("Generated prefix files saved in data/ directory");
    Ok(())
}

fn generate_encoding_data(
    name: &str,
    encoding: &Encoding,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefixes = &encoding.prefixes_of_mergeable_ranks;

    let mut table = HashTableOwned::<PrefixConfig>::with_capacity(prefixes.len(), 87);
    for &prefix in prefixes.iter() {
        table.insert(&prefix, &());
    }

    let bytes = table.raw_bytes();

    let dest_path = Path::new("data").join(format!("{}.prefixes", name));
    let mut f = File::create(&dest_path)?;
    f.write_all(bytes)?;

    println!("Generated {} prefixes saved to {:?}", name, dest_path);
    Ok(())
}
