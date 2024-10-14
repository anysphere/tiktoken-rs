include!("src/load.rs");
include!("src/embedded.rs");
include!("src/rollhash.rs");

use odht::HashTableOwned;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::collections::HashSet;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/embedded.rs");
    println!("cargo:rerun-if-changed=src/load.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/rollhash.rs");

    generate_odht("cl100k_base",
        &load_tiktoken_bpe(
            include_bytes!("data/cl100k_base.tiktoken"),
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        ).unwrap());

    generate_odht("o200k_base",
        &load_tiktoken_bpe(
            include_bytes!("data/o200k_base.tiktoken"),
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        ).unwrap());

    generate_odht("codestral",
        &load_tiktoken_bpe(
            include_bytes!("data/codestral.tiktoken"),
            "bd5e66af07259851e88c3e483f88371dc2408cb0ce8b9787d29eaecdbb78eade",
        ).unwrap());

    generate_odht("llama3",
        &load_tiktoken_bpe(
            include_bytes!("data/llama3.tiktoken"),
            "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
        ).unwrap());

    /*
    generate_odht("deepseekv2",
        &load_tiktoken_bpe(
            include_bytes!("data/deepseekv2.tiktoken"),
            "3516b4e6e24389f7d1b288d861ce063da13296f916d29384e56ea9e0f6ba6674",
        ).unwrap());
    */
}

fn generate_odht(name: &str, mergeable_ranks: &HashMap<Vec<u8>, usize>) {
    let mergeable_ranks_max_key_len = mergeable_ranks
        .keys()
        .map(|bytes| bytes.len())
        .max()
        .unwrap();
    assert!(mergeable_ranks_max_key_len <= MAX_TOKEN_BYTES, "mergeable_ranks_max_key_len ({}) is greater than MAX_TOKEN_BYTES ({})", mergeable_ranks_max_key_len, MAX_TOKEN_BYTES);

    let load_factor = 50; // percent
    let mut encoder = HashTableOwned::<EncoderConfig>::with_capacity(mergeable_ranks.len(), load_factor);
    let mut decoder = HashTableOwned::<DecoderConfig>::with_capacity(mergeable_ranks.len(), load_factor);

    for (k, v) in mergeable_ranks.iter() {
        encoder.insert(k, v);
        decoder.insert(v, k);
    }

    let mut prefixes = mergeable_ranks
        .keys()
        .flat_map(|bytes| {
            (1..=bytes.len())
                .map(|i| roll_hash_slice(&bytes[..i]))
                .collect::<Vec<_>>()
        })
        .collect::<HashSet<_>>();
    prefixes.insert(0);

    let mut prefs = HashTableOwned::<PrefixConfig>::with_capacity(prefixes.len(), load_factor);
    for prefix in prefixes {
        prefs.insert(&prefix, &());
    }

    let dest_path = Path::new(&env::var("OUT_DIR").unwrap()).join(format!("{}.encoder.odht", name));
    let mut f = File::create(&dest_path).unwrap();
    f.write_all(encoder.raw_bytes()).unwrap();

    let dest_path = Path::new(&env::var("OUT_DIR").unwrap()).join(format!("{}.decoder.odht", name));
    let mut f = File::create(&dest_path).unwrap();
    f.write_all(decoder.raw_bytes()).unwrap();

    let dest_path = Path::new(&env::var("OUT_DIR").unwrap()).join(format!("{}.prefix.odht", name));
    let mut f = File::create(&dest_path).unwrap();
    f.write_all(prefs.raw_bytes()).unwrap();

    println!("cargo:rerun-if-changed=data/{}.tiktoken", name);
}
