use std::{
    collections::HashMap,
    fmt::Display,
    io::{self, Cursor},
    path::Path,
};

use models::{RemoteKittenModel, ensure_model_downloaded};
use ndarray::{Array1, Array2, ArrayD, Axis, Ix1, Ix2, s};
use npyz::npz::NpzArchive;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, builder::GraphOptimizationLevel, builder::SessionBuilder},
    value::{DynValue, Tensor},
};
use phonemize::{Phonemizer, PhonemizerBackend};
use preprocess::{TextPreprocessor, basic_english_tokenize, chunk_text};
use thiserror::Error;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;

pub mod models;
pub mod phonemize;
pub mod preprocess;
pub mod wav;

static MODEL: &[u8] = include_bytes!("../model-files/kitten_tts_nano_v0_1.onnx");
static VOICES: &[u8] = include_bytes!("../model-files/voices.npz");

const DEFAULT_CHUNK_MAX_LEN: usize = 400;
const STYLE_DIM: usize = 256;
const TRAILING_TRIM_SAMPLES: usize = 0;

#[derive(Error, Debug, Clone)]
pub enum KittenError {
    #[error("failed to load model: {0}")]
    ModelLoad(String),
    #[error("failed to download model: {0}")]
    ModelDownload(String),
    #[error("failed to execute model: {0}")]
    ModelExecute(String),
    #[error("failed to save model result: {0}")]
    ModelResultSave(String),
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum OrtProvider {
    #[default]
    Auto,
    Cpu,
    #[cfg(feature = "coreml")]
    CoreMl,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(all(feature = "directml", target_os = "windows"))]
    DirectMl,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenerateOptions {
    pub speed: f32,
    pub clean_text: bool,
    pub max_chunk_len: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            clean_text: true,
            max_chunk_len: DEFAULT_CHUNK_MAX_LEN,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum KittenVoice {
    TwoM,
    TwoF,
    ThreeM,
    ThreeF,
    FourM,
    FourF,
    #[default]
    FiveM,
    FiveF,
}

impl Display for KittenVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let voice_str = match self {
            KittenVoice::TwoM => "2-m",
            KittenVoice::TwoF => "2-f",
            KittenVoice::ThreeM => "3-m",
            KittenVoice::ThreeF => "3-f",
            KittenVoice::FourM => "4-m",
            KittenVoice::FourF => "4-f",
            KittenVoice::FiveM => "5-m",
            KittenVoice::FiveF => "5-f",
        };

        write!(f, "expr-voice-{voice_str}")
    }
}

impl KittenVoice {
    pub fn from_model_key(value: &str) -> Option<Self> {
        match value {
            "expr-voice-2-m" => Some(Self::TwoM),
            "expr-voice-2-f" => Some(Self::TwoF),
            "expr-voice-3-m" => Some(Self::ThreeM),
            "expr-voice-3-f" => Some(Self::ThreeF),
            "expr-voice-4-m" => Some(Self::FourM),
            "expr-voice-4-f" => Some(Self::FourF),
            "expr-voice-5-m" => Some(Self::FiveM),
            "expr-voice-5-f" => Some(Self::FiveF),
            _ => None,
        }
    }
}

pub type KittenTokens = HashMap<char, i64>;

#[derive(Debug)]
pub struct KittenModel {
    model: Session,
    voice: Array2<f32>,
    voice_speed_prior: f32,
    phonemizer: Phonemizer,
    preprocessor: TextPreprocessor,
    tokens: KittenTokens,
}

impl KittenModel {
    pub fn get_tokens() -> KittenTokens {
        HashMap::from([
            ('$', 0),
            (';', 1),
            (':', 2),
            (',', 3),
            ('.', 4),
            ('!', 5),
            ('?', 6),
            ('¡', 7),
            ('¿', 8),
            ('—', 9),
            ('…', 10),
            ('"', 11),
            ('«', 12),
            ('»', 13),
            ('"', 14),
            ('"', 15),
            (' ', 16),
            ('A', 17),
            ('B', 18),
            ('C', 19),
            ('D', 20),
            ('E', 21),
            ('F', 22),
            ('G', 23),
            ('H', 24),
            ('I', 25),
            ('J', 26),
            ('K', 27),
            ('L', 28),
            ('M', 29),
            ('N', 30),
            ('O', 31),
            ('P', 32),
            ('Q', 33),
            ('R', 34),
            ('S', 35),
            ('T', 36),
            ('U', 37),
            ('V', 38),
            ('W', 39),
            ('X', 40),
            ('Y', 41),
            ('Z', 42),
            ('a', 43),
            ('b', 44),
            ('c', 45),
            ('d', 46),
            ('e', 47),
            ('f', 48),
            ('g', 49),
            ('h', 50),
            ('i', 51),
            ('j', 52),
            ('k', 53),
            ('l', 54),
            ('m', 55),
            ('n', 56),
            ('o', 57),
            ('p', 58),
            ('q', 59),
            ('r', 60),
            ('s', 61),
            ('t', 62),
            ('u', 63),
            ('v', 64),
            ('w', 65),
            ('x', 66),
            ('y', 67),
            ('z', 68),
            ('ɑ', 69),
            ('ɐ', 70),
            ('ɒ', 71),
            ('æ', 72),
            ('ɓ', 73),
            ('ʙ', 74),
            ('β', 75),
            ('ɔ', 76),
            ('ɕ', 77),
            ('ç', 78),
            ('ɗ', 79),
            ('ɖ', 80),
            ('ð', 81),
            ('ʤ', 82),
            ('ə', 83),
            ('ɘ', 84),
            ('ɚ', 85),
            ('ɛ', 86),
            ('ɜ', 87),
            ('ɝ', 88),
            ('ɞ', 89),
            ('ɟ', 90),
            ('ʄ', 91),
            ('ɡ', 92),
            ('ɠ', 93),
            ('ɢ', 94),
            ('ʛ', 95),
            ('ɦ', 96),
            ('ɧ', 97),
            ('ħ', 98),
            ('ɥ', 99),
            ('ʜ', 100),
            ('ɨ', 101),
            ('ɪ', 102),
            ('ʝ', 103),
            ('ɭ', 104),
            ('ɬ', 105),
            ('ɫ', 106),
            ('ɮ', 107),
            ('ʟ', 108),
            ('ɱ', 109),
            ('ɯ', 110),
            ('ɰ', 111),
            ('ŋ', 112),
            ('ɳ', 113),
            ('ɲ', 114),
            ('ɴ', 115),
            ('ø', 116),
            ('ɵ', 117),
            ('ɸ', 118),
            ('θ', 119),
            ('œ', 120),
            ('ɶ', 121),
            ('ʘ', 122),
            ('ɹ', 123),
            ('ɺ', 124),
            ('ɾ', 125),
            ('ɻ', 126),
            ('ʀ', 127),
            ('ʁ', 128),
            ('ɽ', 129),
            ('ʂ', 130),
            ('ʃ', 131),
            ('ʈ', 132),
            ('ʧ', 133),
            ('ʉ', 134),
            ('ʊ', 135),
            ('ʋ', 136),
            ('ⱱ', 137),
            ('ʌ', 138),
            ('ɣ', 139),
            ('ɤ', 140),
            ('ʍ', 141),
            ('χ', 142),
            ('ʎ', 143),
            ('ʏ', 144),
            ('ʑ', 145),
            ('ʐ', 146),
            ('ʒ', 147),
            ('ʔ', 148),
            ('ʡ', 149),
            ('ʕ', 150),
            ('ʢ', 151),
            ('ǀ', 152),
            ('ǁ', 153),
            ('ǂ', 154),
            ('ǃ', 155),
            ('ˈ', 156),
            ('ˌ', 157),
            ('ː', 158),
            ('ˑ', 159),
            ('ʼ', 160),
            ('ʴ', 161),
            ('ʰ', 162),
            ('ʱ', 163),
            ('ʲ', 164),
            ('ʷ', 165),
            ('ˠ', 166),
            ('ˤ', 167),
            ('˞', 168),
            ('↓', 169),
            ('↑', 170),
            ('→', 171),
            ('↗', 172),
            ('↘', 173),
            ('\'', 174),
            ('̩', 175),
            ('\'', 176),
            ('ᵻ', 177),
        ])
    }

    pub fn model_from_files<P: AsRef<Path>>(
        model_path: P,
        voices_path: P,
        dictionary_path: P,
        voice: KittenVoice,
    ) -> Result<Self, KittenError> {
        Self::model_from_files_with_provider(
            model_path,
            voices_path,
            dictionary_path,
            voice,
            OrtProvider::Auto,
        )
    }

    pub fn model_from_files_with_provider<P: AsRef<Path>>(
        model_path: P,
        voices_path: P,
        dictionary_path: P,
        voice: KittenVoice,
        provider: OrtProvider,
    ) -> Result<Self, KittenError> {
        let model = Self::session_builder(provider)?
            .commit_from_file(model_path)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;

        let mut voices_npz =
            NpzArchive::open(voices_path).map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let phonemizer = Phonemizer::from_file(dictionary_path)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;

        Self::new(voice, &mut voices_npz, model, phonemizer)
    }

    pub fn model_builtin(voice: KittenVoice) -> Result<Self, KittenError> {
        Self::model_builtin_with_provider(voice, OrtProvider::Auto)
    }

    pub fn model_builtin_with_provider(
        voice: KittenVoice,
        provider: OrtProvider,
    ) -> Result<Self, KittenError> {
        let model = Self::session_builder(provider)?
            .commit_from_memory(MODEL)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let mut reader = Cursor::new(VOICES);
        let mut voices_npz =
            NpzArchive::new(&mut reader).map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let phonemizer = Phonemizer::new().map_err(|e| KittenError::ModelLoad(e.to_string()))?;

        Self::new(voice, &mut voices_npz, model, phonemizer)
    }

    pub fn model_latest(voice: KittenVoice) -> Result<Self, KittenError> {
        Self::model_latest_with_provider(voice, OrtProvider::Auto)
    }

    pub fn model_latest_with_provider(
        voice: KittenVoice,
        provider: OrtProvider,
    ) -> Result<Self, KittenError> {
        Self::model_remote_with_provider(voice, provider, RemoteKittenModel::default())
    }

    pub fn model_latest_with_voice_name(
        voice_name: &str,
        provider: OrtProvider,
    ) -> Result<Self, KittenError> {
        Self::model_remote_with_voice_name(voice_name, provider, RemoteKittenModel::default())
    }

    pub fn model_remote_with_provider(
        voice: KittenVoice,
        provider: OrtProvider,
        remote_model: RemoteKittenModel,
    ) -> Result<Self, KittenError> {
        let voice_name = voice.to_string();
        Self::model_remote_with_voice_name(voice_name.as_str(), provider, remote_model)
    }

    pub fn model_remote_with_voice_name(
        voice_name: &str,
        provider: OrtProvider,
        remote_model: RemoteKittenModel,
    ) -> Result<Self, KittenError> {
        let assets = ensure_model_downloaded(remote_model)
            .map_err(|e| KittenError::ModelDownload(e.to_string()))?;
        let resolved_voice = assets.resolve_voice_name(voice_name);
        let voice = KittenVoice::from_model_key(resolved_voice.as_str()).ok_or_else(|| {
            let aliases = assets
                .voice_aliases
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            let voices = assets.available_voices().join(", ");
            KittenError::ModelLoad(format!(
                "unknown voice '{voice_name}'. Available voices: {voices}. Aliases: {aliases}"
            ))
        })?;

        let model = Self::session_builder(provider)?
            .commit_from_file(&assets.model_path)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let mut voices_npz = NpzArchive::open(&assets.voices_path)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let phonemizer = Phonemizer::new().map_err(|e| KittenError::ModelLoad(e.to_string()))?;

        let mut out = Self::new(voice, &mut voices_npz, model, phonemizer)?;
        if let Some(speed_prior) = assets.speed_prior_for(voice_name, resolved_voice.as_str()) {
            out = out.with_speed_prior(speed_prior);
        }

        Ok(out)
    }

    pub fn remote_voice_aliases(
        remote_model: RemoteKittenModel,
    ) -> Result<HashMap<String, String>, KittenError> {
        let assets = ensure_model_downloaded(remote_model)
            .map_err(|e| KittenError::ModelDownload(e.to_string()))?;
        Ok(assets.voice_aliases)
    }

    pub fn new<R: io::Read + io::Seek>(
        voice: KittenVoice,
        npz: &mut NpzArchive<R>,
        model: Session,
        phonemizer: Phonemizer,
    ) -> Result<Self, KittenError> {
        let voice_string = voice.to_string();
        let npy = npz
            .by_name(voice_string.as_str())
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let voice_raw_array = if let Some(voice_raw) = npy {
            voice_raw
        } else {
            return Err(KittenError::ModelLoad(
                "failed to load npy voice file from npz archive".to_string(),
            ));
        };

        let voice_values: Vec<f32> = voice_raw_array
            .data::<f32>()
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?
            .flatten()
            .collect();

        if voice_values.is_empty() || voice_values.len() % STYLE_DIM != 0 {
            return Err(KittenError::ModelLoad(
                "invalid voice embedding shape, expected (N, 256)".to_string(),
            ));
        }

        let rows = voice_values.len() / STYLE_DIM;
        let voice_data = Array2::from_shape_vec((rows, STYLE_DIM), voice_values)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        let tokens = KittenModel::get_tokens();

        Ok(Self {
            model,
            voice: voice_data,
            voice_speed_prior: 1.0,
            phonemizer,
            preprocessor: TextPreprocessor::default(),
            tokens,
        })
    }

    pub fn with_speed_prior(mut self, speed_prior: f32) -> Self {
        self.voice_speed_prior = speed_prior.max(0.01);
        self
    }

    pub fn with_preprocessor(mut self, preprocessor: TextPreprocessor) -> Self {
        self.preprocessor = preprocessor;
        self
    }

    pub fn with_phonemizer_backend(
        mut self,
        backend: PhonemizerBackend,
    ) -> Result<Self, KittenError> {
        self.phonemizer =
            Phonemizer::from_backend(backend).map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        Ok(self)
    }

    pub fn generate(&mut self, text: String) -> Result<(Array1<f32>, Array1<i64>), KittenError> {
        self.generate_with_options(text, GenerateOptions::default())
    }

    pub fn generate_with_options(
        &mut self,
        text: String,
        options: GenerateOptions,
    ) -> Result<(Array1<f32>, Array1<i64>), KittenError> {
        let processed_text = if options.clean_text {
            self.preprocessor.process(text.as_str())
        } else {
            text
        };
        let chunks = chunk_text(processed_text.as_str(), options.max_chunk_len.max(1));
        if chunks.is_empty() {
            return Err(KittenError::ModelExecute(
                "cannot synthesize empty text".to_string(),
            ));
        }

        let mut waveform = Vec::new();
        let mut durations = Vec::new();
        let speed = options.speed.max(0.01) * self.voice_speed_prior;
        for chunk in chunks {
            let (chunk_waveform, chunk_durations) =
                self.generate_single_chunk(chunk.as_str(), speed)?;
            waveform.extend(chunk_waveform.iter().copied());
            durations.extend(chunk_durations.iter().copied());
        }

        Ok((Array1::from(waveform), Array1::from(durations)))
    }

    pub fn generate_from_phonems(
        &mut self,
        phonems: String,
    ) -> Result<(Array1<f32>, Array1<i64>), KittenError> {
        self.generate_from_phonems_with_speed(phonems.as_str(), self.voice_speed_prior, None)
    }

    fn generate_single_chunk(
        &mut self,
        text: &str,
        speed: f32,
    ) -> Result<(Array1<f32>, Array1<i64>), KittenError> {
        let phonemized = self.phonemize_text(text)?;
        self.generate_from_phonems_with_speed(
            phonemized.as_str(),
            speed,
            Some(text.chars().count()),
        )
    }

    fn phonemize_text(&self, text: &str) -> Result<String, KittenError> {
        if self.phonemizer.supports_text_phonemization() {
            if let Some(phonemized) = self
                .phonemizer
                .phonemize_text(text)
                .map_err(|e| KittenError::ModelExecute(e.to_string()))?
            {
                // Match upstream Python path: tokenize phoneme text, then join with spaces.
                return Ok(basic_english_tokenize(phonemized.as_str()).join(" "));
            }
        }

        Ok(basic_english_tokenize(text)
            .into_iter()
            .filter_map(|token| {
                if token.len() == 1 {
                    let ch = token.chars().next()?;
                    if !ch.is_alphanumeric() && self.tokens.contains_key(&ch) {
                        return Some(token);
                    }
                }

                if let Some(phonemized) = self.phonemizer.phonemize(token.as_str()) {
                    return Some(phonemized);
                }

                if token.chars().all(|ch| self.tokens.contains_key(&ch)) {
                    Some(token)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn style_for_text_len(&self, text_len: usize) -> Array2<f32> {
        let row_idx = text_len.min(self.voice.nrows().saturating_sub(1));
        self.voice.slice(s![row_idx..=row_idx, ..]).to_owned()
    }

    fn generate_from_phonems_with_speed(
        &mut self,
        phonems: &str,
        speed: f32,
        style_text_len: Option<usize>,
    ) -> Result<(Array1<f32>, Array1<i64>), KittenError> {
        let mut token_ids: Vec<i64> = phonems
            .chars()
            .filter_map(|c| self.tokens.get(&c))
            .copied()
            .collect();

        if token_ids.is_empty() {
            return Err(KittenError::ModelExecute(
                "phoneme sequence produced zero tokens".to_string(),
            ));
        }

        // Python implementation prepends/appends token id 0.
        token_ids.insert(0, 0);
        token_ids.push(0);
        let text_array: Array1<i64> = Array1::from(token_ids);

        let text_input: Array2<i64> = text_array.insert_axis(Axis(0));
        let text_tensor =
            Tensor::from_array(text_input).map_err(|e| KittenError::ModelExecute(e.to_string()))?;

        let style_len = style_text_len.unwrap_or_else(|| phonems.chars().count());
        let style_tensor = Tensor::from_array(self.style_for_text_len(style_len))
            .map_err(|e| KittenError::ModelExecute(e.to_string()))?;
        let speed_tensor = Tensor::from_array(Array1::from_vec(vec![speed.max(0.01)]))
            .map_err(|e| KittenError::ModelExecute(e.to_string()))?;

        let outputs = self
            .model
            .run(ort::inputs![
            "input_ids" => text_tensor,
            "style" => style_tensor,
            "speed" => speed_tensor
            ])
            .map_err(|e| KittenError::ModelExecute(e.to_string()))?;

        let mut waveform = Self::extract_f32(&outputs["waveform"])?;
        let duration = Self::extract_i64(&outputs["duration"])?;

        if waveform.len() > TRAILING_TRIM_SAMPLES {
            let keep = waveform.len() - TRAILING_TRIM_SAMPLES;
            waveform = waveform.slice(s![..keep]).to_owned();
        }

        Ok((waveform, duration))
    }

    fn extract_f32(value: &DynValue) -> Result<Array1<f32>, KittenError> {
        let array: ArrayD<f32> = value
            .try_extract_array::<f32>()
            .map_err(|e| KittenError::ModelExecute(e.to_string()))?
            .to_owned();

        match array.ndim() {
            1 => array
                .into_dimensionality::<Ix1>()
                .map_err(|e| KittenError::ModelExecute(e.to_string())),
            2 => array
                .into_dimensionality::<Ix2>()
                .map(|a| a.index_axis(Axis(0), 0).to_owned())
                .map_err(|e| KittenError::ModelExecute(e.to_string())),
            _ => Err(KittenError::ModelExecute(
                "unexpected waveform tensor rank".to_string(),
            )),
        }
    }

    fn extract_i64(value: &DynValue) -> Result<Array1<i64>, KittenError> {
        let array: ArrayD<i64> = value
            .try_extract_array::<i64>()
            .map_err(|e| KittenError::ModelExecute(e.to_string()))?
            .to_owned();

        match array.ndim() {
            1 => array
                .into_dimensionality::<Ix1>()
                .map_err(|e| KittenError::ModelExecute(e.to_string())),
            2 => array
                .into_dimensionality::<Ix2>()
                .map(|a| a.index_axis(Axis(0), 0).to_owned())
                .map_err(|e| KittenError::ModelExecute(e.to_string())),
            _ => Err(KittenError::ModelExecute(
                "unexpected duration tensor rank".to_string(),
            )),
        }
    }

    fn session_builder(provider: OrtProvider) -> Result<SessionBuilder, KittenError> {
        let mut builder = Session::builder()
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| KittenError::ModelLoad(e.to_string()))?;

        let providers = Self::execution_providers(provider);
        if !providers.is_empty() {
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| KittenError::ModelLoad(e.to_string()))?;
        }

        Ok(builder)
    }

    fn execution_providers(provider: OrtProvider) -> Vec<ExecutionProviderDispatch> {
        match provider {
            OrtProvider::Auto => {
                #[allow(unused_mut)]
                let mut providers = Vec::new();
                #[cfg(all(feature = "coreml", target_vendor = "apple", target_arch = "aarch64"))]
                {
                    providers.push(CoreMLExecutionProvider::default().build());
                }
                #[cfg(all(feature = "directml", target_os = "windows"))]
                {
                    providers.push(DirectMLExecutionProvider::default().build());
                }
                providers
            }
            OrtProvider::Cpu => Vec::new(),
            #[cfg(feature = "coreml")]
            OrtProvider::CoreMl => vec![CoreMLExecutionProvider::default().build()],
            #[cfg(feature = "cuda")]
            OrtProvider::Cuda => vec![CUDAExecutionProvider::default().build()],
            #[cfg(all(feature = "directml", target_os = "windows"))]
            OrtProvider::DirectMl => vec![DirectMLExecutionProvider::default().build()],
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use crate::wav::save_array1_f32_as_wav;

    use super::*;

    #[test]
    fn model_files() {
        let res = KittenModel::model_from_files(
            "./model-files/kitten_tts_nano_v0_1.onnx",
            "./model-files/voices.npz",
            "./model-files/cmu.dict",
            KittenVoice::default(),
        );
        assert_eq!(res.is_ok(), true);
    }

    #[test]
    fn model_builtin() {
        let res = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(res.is_ok(), true);
    }

    #[test]
    fn generate_from_phonems() {
        let model = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(model.is_ok(), true);
        let res = model.unwrap().generate_from_phonems(
            "ðɪs haɪ kwɔlᵻɾi tiːtiːɛs mɑːdəl wɜːks wɪðaʊt ɐ dʒiːpiːjuː ".to_string(),
        );
        assert_eq!(res.is_ok(), true);
    }

    #[test]
    fn generate() {
        let model = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(model.is_ok(), true);
        let res = model.unwrap().generate(
            "This high quality TTS model works without a GPU. It handles 2026 text cleanly."
                .to_string(),
        );
        assert_eq!(res.is_ok(), true);
    }

    #[test]
    fn generate_phrase_keeps_tts_and_gpu_content() {
        let model = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(model.is_ok(), true);
        let model = model.unwrap();
        let phonemes = model
            .phonemize_text("This high quality TTS model works without a GPU")
            .unwrap();
        assert!(
            phonemes.contains("tiːtiːɛs") || phonemes.contains("tˌiːtˌiːˈɛs"),
            "missing TTS phonemes in: {phonemes}"
        );
        assert!(
            phonemes.contains("dʒiːpiːjuː") || phonemes.contains("dʒˌiːpˌiːjˈuː"),
            "missing GPU phonemes in: {phonemes}"
        );
    }

    #[test]
    fn save() {
        let model = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(model.is_ok(), true);
        let inference = model
            .unwrap()
            .generate("This high quality TTS model works without a GPU".to_string());
        assert_eq!(inference.is_ok(), true);
        let (waveform, _) = inference.unwrap();

        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("out.wav");
        let res = save_array1_f32_as_wav(&waveform, file_path, None);
        assert_eq!(res.is_ok(), true);
    }

    #[test]
    fn save_from_phonems() {
        let model = KittenModel::model_builtin(KittenVoice::default());
        assert_eq!(model.is_ok(), true);
        let inference = model.unwrap().generate_from_phonems(
            "ðɪs haɪ kwɔlᵻɾi tiːtiːɛs mɑːdəl wɜːks wɪðaʊt ɐ dʒiːpiːjuː ".to_string(),
        );
        assert_eq!(inference.is_ok(), true);
        let (waveform, _) = inference.unwrap();

        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("phonems.wav");
        let res = save_array1_f32_as_wav(&waveform, file_path, None);
        assert_eq!(res.is_ok(), true);
    }
}
