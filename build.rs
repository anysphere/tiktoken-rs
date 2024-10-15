include!("src/load.rs");

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/load.rs");
    

    let out_dir = env::var("OUT_DIR").unwrap();
    let phf_path = Path::new(&out_dir).join("static.rs");
    let mut file = File::create(&phf_path).unwrap();
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
    mergeable_ranks: &HashMap<Vec<u8>, usize>,
) {
    writeln!(
        file,
        "    pub const {}: &'static [(&'static [u8], usize)] = &[",
        name.to_uppercase()
    )
    .unwrap();

    let mut keys = mergeable_ranks.keys().collect::<Vec<_>>();
    keys.sort();
    #[cfg(feature = "static-dev")]
    {
        // to keep rust-analyzer and dev-mode experience fast, we generate only the first 10 keys
        keys = keys[..10].to_vec();
    }

    for key in keys {
        let value = mergeable_ranks[key];
        let key_bytes = key
            .iter()
            .map(|byte| format!("\\x{:02x}", byte))
            .collect::<String>();

        writeln!(file, "        (b\"{}\", {}),", key_bytes, value).unwrap();
    }

    writeln!(file, "    ];").unwrap();
}