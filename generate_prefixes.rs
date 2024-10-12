use std::fs::File;
use std::io::Write;
use std::path::Path;
use tiktoken::{Encoding, EncodingFactory};
use phf_codegen::Map;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dest_path = Path::new("src/generated_prefixes.rs");
    let mut f = File::create(&dest_path)?;

    let encodings = [
        ("cl100k", EncodingFactory::cl100k_im()?),
        ("llama3", EncodingFactory::llama3()?),
        ("o200k", EncodingFactory::o200k_im()?),
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

    writeln!(f, "pub static {}_PREFIXES: phf::Map<i64, ()> = ", name.to_uppercase())?;

    let mut map = Map::new();
    for &prefix in prefixes.iter() {
        map.entry(prefix, "()");
    }

    writeln!(f, "{};", map.build())?;
    writeln!(f)?;

    Ok(())
}