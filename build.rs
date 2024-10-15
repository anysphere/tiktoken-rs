include!("src/load.rs");
include!("src/embedded.rs");
include!("src/rollhash.rs");

use odht::HashTableOwned;
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::collections::HashSet;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/embedded.rs");
    println!("cargo:rerun-if-changed=src/load.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/rollhash.rs");

    generate("cl100k_base",
        &load_tiktoken_bpe(
            include_bytes!("data/cl100k_base.tiktoken"),
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        ).unwrap());

    generate("o200k_base",
        &load_tiktoken_bpe(
            include_bytes!("data/o200k_base.tiktoken"),
            "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        ).unwrap());

    generate("codestral",
        &load_tiktoken_bpe(
            include_bytes!("data/codestral.tiktoken"),
            "bd5e66af07259851e88c3e483f88371dc2408cb0ce8b9787d29eaecdbb78eade",
        ).unwrap());

    generate("llama3",
        &load_tiktoken_bpe(
            include_bytes!("data/llama3.tiktoken"),
            "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55",
        ).unwrap());

    /*
    generate("deepseekv2",
        &load_tiktoken_bpe(
            include_bytes!("data/deepseekv2.tiktoken"),
            "3516b4e6e24389f7d1b288d861ce063da13296f916d29384e56ea9e0f6ba6674",
        ).unwrap());
    */
}

fn generate(name: &str, mergeable_ranks: &HashMap<Vec<u8>, usize>) {
    generate_odht(name, mergeable_ranks);
    //generate_phf(name, mergeable_ranks);
    //generate_gperf(name, mergeable_ranks);
}

fn generate_phf(name: &str, mergeable_ranks: &HashMap<Vec<u8>, usize>) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let phf_path = Path::new(&out_dir).join(format!("{}.encoder.phf.rs", name));
    let mut file = File::create(&phf_path).unwrap();

    let mut out = phf_codegen::Map::new();
    for (token, rank) in mergeable_ranks {
        out.entry(token.as_slice(), &rank.to_string());
    }

    write!(
        file,
        "static KEYWORDS: phf::Map<&'static [u8], usize> = {}",
        out.build()
    )
    .unwrap();
    write!(file, ";\n").unwrap();
}


fn generate_gperf(name: &str, mergeable_ranks: &HashMap<Vec<u8>, usize>) {
    let out_dir = env::var("OUT_DIR").unwrap();
    let gperf_path = Path::new(&out_dir).join(format!("{}.gperf", name));
    let mut file = File::create(&gperf_path).unwrap();

    // Write gperf header
    writeln!(file, "%compare-lengths").unwrap();
    writeln!(file, "%language=ANSI-C").unwrap();
    writeln!(file, "%struct-type").unwrap();
    writeln!(file, "%enum").unwrap();
    writeln!(file, "%{{").unwrap();
    writeln!(file, "#include <stddef.h>").unwrap();
    writeln!(file, "%}}").unwrap();
    writeln!(file, "struct TokenInfo {{ const unsigned char *name; size_t len; size_t rank; }};").unwrap();
    writeln!(file, "%%").unwrap();

    // Write token entries
    for (token, rank) in mergeable_ranks {
        let token_str = token.iter().map(|&b| format!("\\x{:02x}", b)).collect::<String>();
        writeln!(file, "{}, {{{}, {}, {}}}", token_str, token_str, token.len(), rank).unwrap();
    }

    // Write gperf footer
    writeln!(file, "%%").unwrap();

    /*
    println!("cargo:rerun-if-changed={}", gperf_path.display());

    // Run gperf to generate C source file
    let output = Command::new("gperf")
        .arg(&gperf_path)
        .arg("--output-file")
        .arg(format!("{}/{}_gperf.c", out_dir, name))
        .arg("-D")  // Handle duplicate hashes
        .arg("-t")  // Include lookup function
        .arg("-L")  // Use ANSI C
        .arg("ANSI-C")
        .arg("-N")  // Function name
        .arg("find_token")
        .arg("-H")  // Hash function name
        .arg("hash_token")
        .output()
        .expect("Failed to execute gperf");

    if !output.status.success() {
        panic!("gperf failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    println!("cargo:rerun-if-changed={}/{}_gperf.c", out_dir, name);
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
