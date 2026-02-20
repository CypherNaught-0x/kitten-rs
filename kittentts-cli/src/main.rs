use std::{
    io::{self, IsTerminal, Read},
    path::PathBuf,
};

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use kittentts_lib::{GenerateOptions, KittenModel, KittenVoice, OrtProvider, wav};

#[derive(Clone, Debug, ValueEnum)]
enum ProviderArg {
    Auto,
    Cpu,
    #[cfg(feature = "coreml")]
    Coreml,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(all(feature = "directml", target_os = "windows"))]
    Directml,
}

impl From<ProviderArg> for OrtProvider {
    fn from(value: ProviderArg) -> Self {
        match value {
            ProviderArg::Auto => OrtProvider::Auto,
            ProviderArg::Cpu => OrtProvider::Cpu,
            #[cfg(feature = "coreml")]
            ProviderArg::Coreml => OrtProvider::CoreMl,
            #[cfg(feature = "cuda")]
            ProviderArg::Cuda => OrtProvider::Cuda,
            #[cfg(all(feature = "directml", target_os = "windows"))]
            ProviderArg::Directml => OrtProvider::DirectMl,
        }
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    text: Option<String>,
    #[arg(short, long, value_name = "OUT_WAV_FILE")]
    wav: PathBuf,
    #[arg(short, long)]
    phonems: bool,
    #[arg(long, value_enum, default_value_t = ProviderArg::Auto)]
    provider: ProviderArg,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long, default_value_t = true)]
    clean_text: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let text = match cli.text {
        Some(text) => text,
        None => {
            if io::stdin().is_terminal() {
                bail!(
                    "No text provided. Either provide text as an argument or pipe it to stdin.\nExample: echo \"hello\" | kittentts-cli --wav out.wav\nExample: kittentts-cli \"hello\" --wav out.wav"
                );
            }
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            let trimmed = buffer.trim();
            if trimmed.is_empty() {
                bail!("No text received");
            }
            trimmed.to_string()
        }
    };

    let mut model =
        KittenModel::model_builtin_with_provider(KittenVoice::default(), cli.provider.into())?;
    let out = if cli.phonems {
        model.generate_from_phonems(text.clone())?
    } else {
        model.generate_with_options(
            text.clone(),
            GenerateOptions {
                speed: cli.speed,
                clean_text: cli.clean_text,
                ..GenerateOptions::default()
            },
        )?
    };
    wav::save_array1_f32_as_wav(&out.0, cli.wav, None)?;

    println!("Finished!");
    Ok(())
}
