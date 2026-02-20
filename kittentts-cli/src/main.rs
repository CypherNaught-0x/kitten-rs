use std::{
    io::{self, IsTerminal, Read},
    path::PathBuf,
};

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use kittentts_lib::{
    GenerateOptions, KittenModel, OrtProvider,
    models::{RemoteKittenModel, ensure_model_downloaded},
    phonemize::PhonemizerBackend,
    wav,
};

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

#[derive(Clone, Debug, ValueEnum)]
enum ModelArg {
    NanoInt8,
    NanoFp32,
    Micro,
    Mini,
}

impl From<ModelArg> for RemoteKittenModel {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::NanoInt8 => RemoteKittenModel::NanoInt8,
            ModelArg::NanoFp32 => RemoteKittenModel::NanoFp32,
            ModelArg::Micro => RemoteKittenModel::Micro,
            ModelArg::Mini => RemoteKittenModel::Mini,
        }
    }
}

#[derive(Clone, Debug, ValueEnum)]
enum PhonemizerArg {
    Rust,
    Espeakng,
    Misaki,
}

impl From<PhonemizerArg> for PhonemizerBackend {
    fn from(value: PhonemizerArg) -> Self {
        match value {
            PhonemizerArg::Rust => PhonemizerBackend::Rust,
            PhonemizerArg::Espeakng => PhonemizerBackend::EspeakNg,
            PhonemizerArg::Misaki => PhonemizerBackend::Misaki,
        }
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    text: Option<String>,
    #[arg(short, long, value_name = "OUT_WAV_FILE")]
    wav: Option<PathBuf>,
    #[arg(short, long)]
    phonems: bool,
    #[arg(long, value_enum, default_value_t = ProviderArg::Auto)]
    provider: ProviderArg,
    #[arg(long, default_value_t = 1.0)]
    speed: f32,
    #[arg(long, default_value_t = true)]
    clean_text: bool,
    #[arg(long, value_enum, default_value_t = ModelArg::NanoInt8)]
    model: ModelArg,
    #[arg(long, default_value = "expr-voice-5-m")]
    voice: String,
    #[arg(long)]
    list_voices: bool,
    #[arg(long, value_enum, default_value_t = PhonemizerArg::Rust)]
    phonemizer: PhonemizerArg,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let remote_model: RemoteKittenModel = cli.model.into();

    if cli.list_voices {
        let assets = ensure_model_downloaded(remote_model)?;
        println!("Voices:");
        for voice in assets.available_voices() {
            println!("  {voice}");
        }
        if !assets.voice_aliases.is_empty() {
            println!("\nVoice aliases:");
            let mut aliases: Vec<_> = assets.voice_aliases.iter().collect();
            aliases.sort_by(|a, b| a.0.cmp(b.0));
            for (alias, voice) in aliases {
                println!("  {alias} -> {voice}");
            }
        }
        return Ok(());
    }

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
        KittenModel::model_remote_with_voice_name(&cli.voice, cli.provider.into(), remote_model)?
            .with_phonemizer_backend(cli.phonemizer.into())?;
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
    let out_wav = cli.wav.ok_or_else(|| {
        anyhow::anyhow!(
            "Missing required --wav output path.\nExample: kittentts-cli \"hello\" --wav out.wav"
        )
    })?;
    wav::save_array1_f32_as_wav(&out.0, out_wav, None)?;

    println!("Finished!");
    Ok(())
}
