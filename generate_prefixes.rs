use std::fs::File;
use std::io::Write;
use std::path::Path;
use tiktoken::{Encoding, EncodingFactory};
use phf_codegen::Set;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dest_path = Path::new("src/generated_prefixes.rs");
    let mut f = File::create(&dest_path)?;

    let encodings = [
        ("cl100k_base", EncodingFactory::cl100k_base()?),
        ("llama3", EncodingFactory::llama3()?),
        ("o200k_base", EncodingFactory::o200k_base()?),
        ("codestral", EncodingFactory::codestral()?),
    ];

    writeln!(f, "// This file is auto-generated. Do not edit manually.")?;
    writeln!(f)?;

    for (name, encoding) in encodings.iter() {
        generate_encoding_data(&mut f, name, encoding)?;
    }

    println!("Generated prefixes saved to {:?}", dest_path);
    Ok(())
}

fn generate_encoding_data(
    f: &mut File,
    name: &str,
    encoding: &Encoding,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefixes = &encoding.prefixes_of_mergeable_ranks;

    writeln!(f, "pub static {}_PREFIXES: phf::Set<i64> = ", name.to_uppercase())?;

    let mut set = Set::new();
    for &prefix in prefixes.iter() {
        set.entry(prefix);
    }

    writeln!(f, "{};", set.build())?;
    writeln!(f)?;

    Ok(())
}