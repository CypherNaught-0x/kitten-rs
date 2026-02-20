# Kitten TTS Rust Port

Rust port of [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS) with library and CLI support.

## Highlights

- Auto-download + cache of newest `0.8` Hugging Face models.
- Default model is `nano-int8` (`KittenML/kitten-tts-nano-0.8-int8`).
- CLI model selection: `nano-int8`, `nano-fp32`, `micro`, `mini`.
- Voice aliases from model `config.json` are supported in both library and CLI.
- `speed_priors` from model `config.json` are applied automatically per selected voice.
- Additional phonemizers:
  - `rust` (CMUdict + direct references to `languages/en/en_list`, `languages/en/en_rules`, `languages/en/en_emoji`)
  - `espeakng` (native `espeak-ng`/`espeak` binary)
  - `misaki` (Python `misaki`)
- Execution provider selection:
  - `auto`
  - `cpu`
  - `coreml` (feature `coreml`)
  - `cuda` (feature `cuda`)
  - `directml` (feature `directml`, Windows only)

## CLI

```bash
# Default: downloads/uses newest 0.8 nano-int8 model
cargo run -p kittentts-cli -- "This high quality TTS model works without a GPU" --wav out.wav

# Pick a specific model
cargo run -p kittentts-cli -- "hello" --wav out.wav --model mini

# Use a voice alias from config.json
cargo run -p kittentts-cli -- "hello" --wav out.wav --voice Jasper

# List available canonical voices and aliases for the selected model
cargo run -p kittentts-cli -- --model nano-int8 --list-voices

# Select phonemizer backend
cargo run -p kittentts-cli -- "This high quality TTS model works without a GPU" --wav out.wav --phonemizer espeakng

# Use explicit execution provider
cargo run -p kittentts-cli -- "hello" --wav out.wav --provider cpu

# Speed + preprocessing controls
cargo run -p kittentts-cli -- "GPT-3 is 50% faster in 2026" --wav out.wav --speed 1.05 --clean-text true
```

### CoreML (Apple Silicon)

```bash
cargo run -p kittentts-cli --features coreml -- "hello from m-series" --wav out.wav --provider coreml
```

### CUDA (optional)

```bash
cargo run -p kittentts-cli --features cuda -- "hello from cuda" --wav out.wav --provider cuda
```

### DirectML (Windows, optional)

```bash
cargo run -p kittentts-cli --features directml -- "hello from directml" --wav out.wav --provider directml
```

## Library usage

```rust
use kittentts_lib::{GenerateOptions, KittenModel, OrtProvider, wav};
use kittentts_lib::phonemize::PhonemizerBackend;

let mut model = KittenModel::model_latest_with_voice_name("Jasper", OrtProvider::Auto)?
    .with_phonemizer_backend(PhonemizerBackend::EspeakNg)?;
let (waveform, _duration) = model.generate_with_options(
    "This high quality TTS model works without a GPU in 2026".to_string(),
    GenerateOptions::default(),
)?;
wav::save_array1_f32_as_wav(&waveform, "out.wav", Some(24000))?;
// Ok::<(), Box<dyn std::error::Error>>(())
```

```rust
use kittentts_lib::{KittenModel, models::RemoteKittenModel};

let aliases = KittenModel::remote_voice_aliases(RemoteKittenModel::NanoInt8)?;
// e.g. Jasper -> expr-voice-5-m
# Ok::<(), Box<dyn std::error::Error>>(())
```

The `misaki` backend requires Python packages installed in your environment.
The `espeakng` backend requires `espeak-ng` or `espeak` available in `PATH`.
WAV output defaults to `24000` Hz (matching upstream Python examples).

## License

This repository is licensed under GNU GPLv3 (or later). See `LICENSE`.
