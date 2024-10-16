include!("src/load.rs");
include!("src/rollhash.rs");
include!("src/odht.rs");

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use odht::HashTableOwned;
use rustc_hash::FxHashSet as HashSet;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/load.rs");
    println!("cargo:rerun-if-changed=src/rollhash.rs");
    println!("cargo:rerun-if-changed=src/odht.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let mut file = File::create(&Path::new(&out_dir).join("odht_gen.rs")).unwrap();
    writeln!(file, "pub mod data {{").unwrap();

    generate("r50k_base",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/r50k_base.tiktoken"),
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
        ).unwrap());

    generate("p50k_base",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/p50k_base.tiktoken"),
            "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        ).unwrap());

    generate("cl100k_base",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/cl100k_base.tiktoken"),
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        ).unwrap());

    generate("o200k_base",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/o200k_base.tiktoken"),
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        ).unwrap());

    generate("codestral",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/codestral.tiktoken"),
            "bd5e66af07259851e88c3e483f88371dc2408cb0ce8b9787d29eaecdbb78eade",
        ).unwrap());

    generate("llama3",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/llama3.tiktoken"),
            "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
        ).unwrap());

    generate("deepseekv2",
        &mut file,
        &load_tiktoken_bpe(
            include_bytes!("data/deepseekv2.tiktoken"),
            "3516b4e6e24389f7d1b288d861ce063da13296f916d29384e56ea9e0f6ba6674",
        ).unwrap());

    writeln!(file, "}}").unwrap();
}

fn generate(
    name: &str,
    file: &mut File,
    mergeable_ranks: &HashMap<Vec<u8>, Rank>,
) {
    writeln!(
        file,
        "    pub const {}_PREFIXES_ODHT: &'static [u8] = include_bytes!(\"{}.prefixes.odht\");",
        name.to_uppercase(),
        name
    ).unwrap();

    let mut prefixes_of_mergeable_ranks = mergeable_ranks
    .keys()
    .flat_map(|bytes| {
        (1..=bytes.len())
            .map(|i| roll_hash_slice(&bytes[..i]))
            .collect::<Vec<_>>()
    })
    .collect::<HashSet<_>>();
    prefixes_of_mergeable_ranks.insert(0);
    prefixes_of_mergeable_ranks.shrink_to_fit();

    let mut odht = HashTableOwned::<PrefixConfig>::with_capacity(prefixes_of_mergeable_ranks.len(), 50);
    for prefix in prefixes_of_mergeable_ranks {
        odht.insert(&prefix, &());
    }

    let mut file = File::create(format!("{}/{}.prefixes.odht", env::var("OUT_DIR").unwrap(), name)).unwrap();
    file.write_all(odht.raw_bytes()).unwrap();
}
