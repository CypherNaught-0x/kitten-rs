# Kitten TTS Rust Port

Rust port of [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS) with a library crate and CLI.

## CLI

```bash
# CPU/default build
cargo run -p kittentts-cli -- "This high quality TTS model works without a GPU" --wav out.wav

# Use explicit provider
cargo run -p kittentts-cli -- "hello" --wav out.wav --provider cpu

# Control speed and text preprocessing
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
use kittentts_lib::{GenerateOptions, KittenModel, KittenVoice, OrtProvider, wav};

let mut model = KittenModel::model_builtin_with_provider(KittenVoice::default(), OrtProvider::Auto)?;
let (waveform, _duration) = model.generate_with_options(
    "This high quality TTS model works without a GPU in 2026".to_string(),
    GenerateOptions::default(),
)?;
wav::save_array1_f32_as_wav(&waveform, "out.wav", Some(24000))?;
// Ok::<(), Box<dyn std::error::Error>>(())
```
