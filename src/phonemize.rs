use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::Path,
    process::{Command, Stdio},
    str::FromStr,
};

use crate::preprocess::basic_english_tokenize;
use cmudict_fast::{Cmudict, Rule};
use thiserror::Error;

const DICT: &str = include_str!("../model-files/cmu.dict");
const ESPEAK_EN_LIST: &str = include_str!("../languages/en/en_list");
const ESPEAK_EN_RULES: &str = include_str!("../languages/en/en_rules");
const ESPEAK_EN_EMOJI: &str = include_str!("../languages/en/en_emoji");

const MISAKI_CHECK_SCRIPT: &str = r#"
import importlib
importlib.import_module("misaki")
from misaki import en
"#;

const MISAKI_PHONEMIZE_SCRIPT: &str = r#"
import sys
from misaki import en

def _to_str(v):
    if isinstance(v, str):
        return v
    if isinstance(v, tuple):
        for x in v:
            if isinstance(x, str):
                return x
    if hasattr(v, "phonemes"):
        p = getattr(v, "phonemes")
        if isinstance(p, str):
            return p
    return ""

text = sys.stdin.read()
if not text:
    sys.exit(0)

g2p = None
for kwargs in (
    {"trf": False, "british": False},
    {"trf": False},
    {}
):
    try:
        g2p = en.G2P(**kwargs)
        break
    except TypeError:
        pass

if g2p is None:
    g2p = en.G2P()

out = _to_str(g2p(text))
if (not out) and hasattr(g2p, "phonemize"):
    out = _to_str(g2p.phonemize(text))
if not out:
    raise RuntimeError("misaki returned unsupported output format")
sys.stdout.write(out)
"#;

#[derive(Error, Debug, Clone)]
pub enum PhonemizerError {
    #[error("failed to load dictionary: {0}")]
    DictLoad(String),
    #[error("backend is not available: {0}")]
    BackendUnavailable(String),
    #[error("backend execution failed: {0}")]
    BackendExecute(String),
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum PhonemizerBackend {
    #[default]
    Rust,
    EspeakNg,
    Misaki,
}

#[derive(Debug)]
enum Backend {
    Rust(RustPhonemizer),
    Espeak(EspeakPhonemizer),
    Python(PythonPhonemizer),
}

#[derive(Debug)]
pub struct Phonemizer {
    backend: Backend,
}

#[derive(Debug)]
struct RustPhonemizer {
    dict: Cmudict,
    ipa: HashMap<&'static str, &'static str>,
    language_data: EspeakLanguageData,
    espeak_fallback: Option<EspeakPhonemizer>,
}

#[derive(Debug)]
struct PythonPhonemizer {
    backend: PhonemizerBackend,
    python_bin: String,
}

#[derive(Debug)]
struct EspeakPhonemizer {
    espeak_bin: String,
}

#[derive(Debug)]
struct EspeakLanguageData {
    replacements: Vec<(String, String)>,
    emoji_names: HashMap<String, String>,
    list_words: HashSet<String>,
}

impl EspeakLanguageData {
    fn load() -> Self {
        Self {
            replacements: parse_en_rules_replacements(ESPEAK_EN_RULES),
            emoji_names: parse_en_emoji_names(ESPEAK_EN_EMOJI),
            list_words: parse_en_list_words(ESPEAK_EN_LIST),
        }
    }

    fn apply_replacements(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (from, to) in &self.replacements {
            if from.is_empty() || from == to {
                continue;
            }
            out = out.replace(from, to);
        }
        out
    }
}

impl Phonemizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PhonemizerError> {
        let dict = Cmudict::new(path).map_err(|e| PhonemizerError::DictLoad(e.to_string()))?;
        let language_data = EspeakLanguageData::load();
        Ok(Self {
            backend: Backend::Rust(RustPhonemizer {
                dict,
                ipa: get_ipa(),
                language_data,
                espeak_fallback: EspeakPhonemizer::new().ok(),
            }),
        })
    }

    pub fn new() -> Result<Self, PhonemizerError> {
        let dict = Cmudict::from_str(DICT).map_err(|e| PhonemizerError::DictLoad(e.to_string()))?;
        let language_data = EspeakLanguageData::load();
        Ok(Self {
            backend: Backend::Rust(RustPhonemizer {
                dict,
                ipa: get_ipa(),
                language_data,
                espeak_fallback: EspeakPhonemizer::new().ok(),
            }),
        })
    }

    pub fn from_backend(backend: PhonemizerBackend) -> Result<Self, PhonemizerError> {
        match backend {
            PhonemizerBackend::Rust => Self::new(),
            PhonemizerBackend::EspeakNg => Ok(Self {
                backend: Backend::Espeak(EspeakPhonemizer::new()?),
            }),
            PhonemizerBackend::Misaki => {
                let python = PythonPhonemizer::new(backend)?;
                Ok(Self {
                    backend: Backend::Python(python),
                })
            }
        }
    }

    pub fn backend(&self) -> PhonemizerBackend {
        match self.backend {
            Backend::Rust(_) => PhonemizerBackend::Rust,
            Backend::Espeak(_) => PhonemizerBackend::EspeakNg,
            Backend::Python(ref p) => p.backend,
        }
    }

    pub fn supports_text_phonemization(&self) -> bool {
        match &self.backend {
            Backend::Rust(rust) => rust.supports_text_phonemization(),
            Backend::Espeak(_) | Backend::Python(_) => true,
        }
    }

    pub fn phonemize_text(&self, text: &str) -> Result<Option<String>, PhonemizerError> {
        match &self.backend {
            Backend::Rust(rust) => rust.phonemize_text(text),
            Backend::Espeak(espeak) => {
                Ok(Some(espeak.phonemize_text(text)?).filter(|s| !s.is_empty()))
            }
            Backend::Python(python) => {
                Ok(Some(python.phonemize_text(text)?).filter(|s| !s.is_empty()))
            }
        }
    }

    pub fn phonemize(&self, word: &str) -> Option<String> {
        match &self.backend {
            Backend::Rust(rust) => rust.phonemize(word),
            Backend::Espeak(espeak) => espeak.phonemize_text(word).ok().filter(|s| !s.is_empty()),
            Backend::Python(python) => python.phonemize_text(word).ok().filter(|s| !s.is_empty()),
        }
    }
}

impl EspeakPhonemizer {
    fn new() -> Result<Self, PhonemizerError> {
        let candidates = ["espeak-ng", "espeak"];
        for candidate in candidates {
            if Command::new(candidate)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok()
            {
                return Ok(Self {
                    espeak_bin: candidate.to_string(),
                });
            }
        }

        Err(PhonemizerError::BackendUnavailable(
            "could not find `espeak-ng` or `espeak` in PATH".to_string(),
        ))
    }

    fn phonemize_text(&self, text: &str) -> Result<String, PhonemizerError> {
        let mut child = Command::new(&self.espeak_bin)
            .args(["-v", "en-us", "--ipa", "-q", "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PhonemizerError::BackendUnavailable(e.to_string()))?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| PhonemizerError::BackendExecute(e.to_string()))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| PhonemizerError::BackendExecute(e.to_string()))?;

        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(PhonemizerError::BackendUnavailable(if stderr.is_empty() {
            format!("{} exited with {}", self.espeak_bin, output.status)
        } else {
            stderr
        }))
    }
}

impl PythonPhonemizer {
    fn new(backend: PhonemizerBackend) -> Result<Self, PhonemizerError> {
        let phonemizer = Self {
            backend,
            python_bin: "python3".to_string(),
        };
        phonemizer.check_available()?;
        Ok(phonemizer)
    }

    fn check_available(&self) -> Result<(), PhonemizerError> {
        let script = match self.backend {
            PhonemizerBackend::Misaki => MISAKI_CHECK_SCRIPT,
            PhonemizerBackend::Rust | PhonemizerBackend::EspeakNg => return Ok(()),
        };
        self.run_script(script, "")?;
        Ok(())
    }

    fn phonemize_text(&self, text: &str) -> Result<String, PhonemizerError> {
        let script = match self.backend {
            PhonemizerBackend::Misaki => MISAKI_PHONEMIZE_SCRIPT,
            PhonemizerBackend::Rust | PhonemizerBackend::EspeakNg => return Ok(text.to_string()),
        };
        self.run_script(script, text)
    }

    fn run_script(&self, script: &str, text: &str) -> Result<String, PhonemizerError> {
        let mut child = Command::new(&self.python_bin)
            .arg("-c")
            .arg(script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PhonemizerError::BackendUnavailable(e.to_string()))?;

        if let Some(stdin) = child.stdin.as_mut() {
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| PhonemizerError::BackendExecute(e.to_string()))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| PhonemizerError::BackendExecute(e.to_string()))?;
        if output.status.success() {
            return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(PhonemizerError::BackendUnavailable(if stderr.is_empty() {
            format!("{} exited with {}", self.python_bin, output.status)
        } else {
            stderr
        }))
    }
}

impl RustPhonemizer {
    fn supports_text_phonemization(&self) -> bool {
        self.espeak_fallback.is_some()
    }

    fn phonemize_text(&self, text: &str) -> Result<Option<String>, PhonemizerError> {
        let Some(espeak) = &self.espeak_fallback else {
            return Ok(None);
        };
        // Keep sentence-level parity with native espeak output.
        let normalized = self.language_data.apply_replacements(text);
        let out = espeak.phonemize_text(normalized.as_str())?;
        Ok(Some(out).filter(|s| !s.is_empty()))
    }

    fn phonemize_phrase(&self, phrase: &str) -> String {
        basic_english_tokenize(phrase)
            .into_iter()
            .filter_map(|token| {
                if token.len() == 1 {
                    let ch = token.chars().next()?;
                    if !ch.is_alphanumeric() {
                        return Some(token);
                    }
                }
                self.phonemize(token.as_str()).or(Some(token))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn fallback_espeak_word(&self, text: &str) -> Option<String> {
        let espeak = self.espeak_fallback.as_ref()?;
        espeak.phonemize_text(text).ok().filter(|s| !s.is_empty())
    }

    fn phonemize(&self, word: &str) -> Option<String> {
        let normalized_word = self.language_data.apply_replacements(word);
        if let Some(expansion) = self.language_data.emoji_names.get(&normalized_word) {
            return Some(self.phonemize_phrase(expansion));
        }

        let lower_case = normalized_word.to_lowercase();
        let upper_case = normalized_word.to_uppercase();

        if self.language_data.list_words.contains(&lower_case) {
            if let Some(espeak) = self.fallback_espeak_word(normalized_word.as_str()) {
                return Some(espeak);
            }
        }

        let rules = self.dict.get(lower_case.as_str());
        let rule = if let Some(rule) = rules {
            rule[0].clone()
        } else {
            if lower_case.len() <= 4 && lower_case.chars().all(|ch| ch.is_ascii_alphabetic()) {
                return Some(spell_ascii_word(lower_case.as_str()));
            }
            let rule_from_str = Rule::from_str(upper_case.as_str());
            match rule_from_str {
                Ok(rule) => rule,
                Err(_) => return None,
            }
        };

        let pronunciation = rule.pronunciation();
        let phonemized: String = if pronunciation.is_empty() {
            upper_case
        } else {
            pronunciation
                .iter()
                .filter_map(|p| {
                    let raw = p.to_string();
                    let stress = if raw.ends_with('1') {
                        Some("ˈ")
                    } else if raw.ends_with('2') {
                        Some("ˌ")
                    } else {
                        None
                    };
                    let key = raw.trim_end_matches(['0', '1', '2']);
                    let ipa = self
                        .ipa
                        .get(key)
                        .copied()
                        .or_else(|| self.ipa.get(raw.as_str()).copied())?;

                    if let Some(stress_mark) = stress.filter(|_| is_arpabet_vowel(key)) {
                        Some(format!("{stress_mark}{ipa}"))
                    } else {
                        Some(ipa.to_string())
                    }
                })
                .collect()
        };

        if phonemized.is_empty() {
            self.fallback_espeak_word(normalized_word.as_str())
        } else {
            Some(phonemized)
        }
    }
}

fn parse_en_rules_replacements(source: &str) -> Vec<(String, String)> {
    let mut in_replace = false;
    let mut replacements = Vec::new();

    for raw_line in source.lines() {
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == ".replace" {
            in_replace = true;
            continue;
        }
        if !in_replace {
            continue;
        }
        if line.starts_with(".group") || line.starts_with(".") {
            break;
        }

        let mut parts = line.split_whitespace();
        let Some(from) = parts.next() else {
            continue;
        };
        let Some(to) = parts.next() else {
            continue;
        };
        replacements.push((from.to_string(), to.to_string()));
    }

    replacements.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    replacements
}

fn parse_en_emoji_names(source: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();

    for raw_line in source.lines() {
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('$') {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(symbol) = parts.next() else {
            continue;
        };
        let name = parts.collect::<Vec<_>>().join(" ");
        if symbol.is_empty() || name.is_empty() {
            continue;
        }

        out.entry(symbol.to_string()).or_insert(name);
    }

    out
}

fn parse_en_list_words(source: &str) -> HashSet<String> {
    let mut out = HashSet::new();

    for raw_line in source.lines() {
        let line = raw_line.split("//").next().unwrap_or("").trim();
        if line.is_empty()
            || line.starts_with('.')
            || line.starts_with('?')
            || line.starts_with('$')
            || line.starts_with('<')
        {
            continue;
        }

        let mut parts = line.split_whitespace();
        let Some(word) = parts.next() else {
            continue;
        };
        if !word.chars().all(|ch| ch.is_ascii_lowercase()) {
            continue;
        }

        if parts.next().is_none() {
            continue;
        }
        out.insert(word.to_string());
    }

    out
}

fn get_ipa() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("AA", "ɑ"),
        ("AA1", "ɑː"),
        ("AA2", "ɑː"),
        ("AE", "æ"),
        ("AE1", "æ"),
        ("AE2", "æ"),
        ("AH", "ə"),
        ("AH1", "ʌ"),
        ("AH2", "ə"),
        ("AO", "ɔ"),
        ("AO1", "ɔː"),
        ("AO2", "ɔː"),
        ("AW", "aʊ"),
        ("AW1", "aʊ"),
        ("AW2", "aʊ"),
        ("AY", "aɪ"),
        ("AY1", "aɪ"),
        ("AY2", "aɪ"),
        ("EH", "ɛ"),
        ("EH1", "ɛ"),
        ("EH2", "ɛ"),
        ("ER", "ɝ"),
        ("ER1", "ɝː"),
        ("ER2", "ɝː"),
        ("EY", "eɪ"),
        ("EY1", "eɪ"),
        ("EY2", "eɪ"),
        ("IH", "ᵻ"),
        ("IH1", "ɪ"),
        ("IH2", "ɪ"),
        ("IY", "i"),
        ("IY1", "iː"),
        ("IY2", "iː"),
        ("OW", "oʊ"),
        ("OW1", "oʊ"),
        ("OW2", "oʊ"),
        ("OY", "ɔɪ"),
        ("OY1", "ɔɪ"),
        ("OY2", "ɔɪ"),
        ("UH", "ʊ"),
        ("UH1", "ʊ"),
        ("UH2", "ʊ"),
        ("UW", "u"),
        ("UW1", "uː"),
        ("UW2", "uː"),
        ("B", "b"),
        ("CH", "tʃ"),
        ("D", "d"),
        ("DH", "ð"),
        ("F", "f"),
        ("G", "ɡ"),
        ("HH", "h"),
        ("JH", "dʒ"),
        ("K", "k"),
        ("L", "l"),
        ("M", "m"),
        ("N", "n"),
        ("NG", "ŋ"),
        ("P", "p"),
        ("R", "ɹ"),
        ("S", "s"),
        ("SH", "ʃ"),
        ("T", "t"),
        ("TH", "θ"),
        ("V", "v"),
        ("W", "w"),
        ("Y", "j"),
        ("Z", "z"),
        ("ZH", "ʒ"),
    ])
}

fn spell_ascii_word(word: &str) -> String {
    word.chars()
        .filter_map(letter_name_ipa)
        .collect::<Vec<_>>()
        .join("")
}

fn letter_name_ipa(ch: char) -> Option<&'static str> {
    match ch.to_ascii_lowercase() {
        'a' => Some("eɪ"),
        'b' => Some("biː"),
        'c' => Some("siː"),
        'd' => Some("diː"),
        'e' => Some("iː"),
        'f' => Some("ɛf"),
        'g' => Some("dʒiː"),
        'h' => Some("eɪtʃ"),
        'i' => Some("aɪ"),
        'j' => Some("dʒeɪ"),
        'k' => Some("keɪ"),
        'l' => Some("ɛl"),
        'm' => Some("ɛm"),
        'n' => Some("ɛn"),
        'o' => Some("oʊ"),
        'p' => Some("piː"),
        'q' => Some("kjuː"),
        'r' => Some("ɑɹ"),
        's' => Some("ɛs"),
        't' => Some("tiː"),
        'u' => Some("juː"),
        'v' => Some("viː"),
        'w' => Some("dʌbəljuː"),
        'x' => Some("ɛks"),
        'y' => Some("waɪ"),
        'z' => Some("ziː"),
        _ => None,
    }
}

fn is_arpabet_vowel(phone: &str) -> bool {
    matches!(
        phone,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(test)]
mod tests {
    use super::{Phonemizer, PhonemizerBackend};
    use crate::preprocess::{TextPreprocessor, basic_english_tokenize};

    const COMPARISON_CORPUS: &[&str] = &[
        "This high quality TTS model works without a GPU.",
        "The quick brown fox jumps over 13 lazy dogs.",
        "Model version 0.8 ships on 2026-02-20.",
        "CPU, GPU, and TTS must remain intelligible.",
        "I paid $12.50 for 3 items with 20% off.",
        "Set value to -3.14, then add +42.",
        "Email support@kitten.ai or visit https://kitten.ai/docs.",
        "NVIDIA RTX 4090 beats older cards in many benchmarks.",
        "Use C++, C#, and Rust in one project.",
        "Can you read 1st, 2nd, and 3rd correctly?",
        "Call 555-123-4567 before 10:30 a.m.",
        "Temperature range is -5C to 37C today.",
        "The SKU is AB-1234-XZ and ID is Q9Z.",
        "I owe you £5, €7, and ¥900.",
        "We need 99.99% uptime, no excuses.",
        "Edge-cases include emojis :) and symbols #!&*.",
        "Dr. Smith lives on 5th Ave., Apt #12.",
        "JSON keys: user_id, created_at, and is_admin.",
        "Read RFC-8259 and ISO-8601 carefully.",
        "Mixing CAPS and lowerCase can break G2P.",
    ];

    fn maybe_espeak_phonemizer() -> Option<Phonemizer> {
        match Phonemizer::from_backend(PhonemizerBackend::EspeakNg) {
            Ok(p) => Some(p),
            Err(err) => {
                eprintln!("skipping espeak comparison tests: {err}");
                None
            }
        }
    }

    fn rust_sentence_phonemes(phonemizer: &Phonemizer, sentence: &str) -> String {
        if phonemizer.supports_text_phonemization()
            && let Ok(Some(phonemized)) = phonemizer.phonemize_text(sentence)
        {
            return basic_english_tokenize(phonemized.as_str()).join(" ");
        }

        basic_english_tokenize(sentence)
            .into_iter()
            .filter_map(|token| {
                if token.len() == 1 {
                    let ch = token.chars().next()?;
                    if !ch.is_alphanumeric() {
                        return Some(token);
                    }
                }
                if let Some(phoneme) = phonemizer.phonemize(token.as_str()) {
                    return Some(phoneme);
                }
                Some(token)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn espeak_sentence_phonemes(phonemizer: &Phonemizer, sentence: &str) -> String {
        let phonemized = phonemizer
            .phonemize_text(sentence)
            .expect("espeak backend should produce phonemes")
            .expect("espeak backend returned empty output");
        basic_english_tokenize(phonemized.as_str()).join(" ")
    }

    #[test]
    fn acronym_tts_is_spelled_out() {
        let p = Phonemizer::new().unwrap();
        let out = p.phonemize("tts").unwrap();
        assert_eq!(out, "tiːtiːɛs");
    }

    #[test]
    fn acronym_gpu_is_spelled_out() {
        let p = Phonemizer::new().unwrap();
        let out = p.phonemize("gpu").unwrap();
        assert_eq!(out, "dʒiːpiːjuː");
    }

    #[test]
    fn rust_backend_is_default() {
        let p = Phonemizer::new().unwrap();
        assert_eq!(p.backend(), PhonemizerBackend::Rust);
    }

    #[test]
    fn compare_rust_and_espeak_cleaned_corpus() {
        let Some(espeak) = maybe_espeak_phonemizer() else {
            return;
        };
        let rust = Phonemizer::new().unwrap();
        let preprocessor = TextPreprocessor::default();

        for sentence in COMPARISON_CORPUS {
            let cleaned = preprocessor.process(sentence);
            if cleaned.is_empty() {
                continue;
            }

            let rust_out = rust_sentence_phonemes(&rust, cleaned.as_str());
            let espeak_out = espeak_sentence_phonemes(&espeak, cleaned.as_str());

            assert!(
                !rust_out.trim().is_empty(),
                "rust output is empty for: {cleaned}"
            );
            assert!(
                !espeak_out.trim().is_empty(),
                "espeak output is empty for: {cleaned}"
            );
            assert_eq!(
                rust_out, espeak_out,
                "cleaned corpus mismatch for input={cleaned:?}\nrust={rust_out:?}\nespeak={espeak_out:?}"
            );
        }
    }

    #[test]
    fn compare_rust_and_espeak_raw_corpus() {
        let Some(espeak) = maybe_espeak_phonemizer() else {
            return;
        };
        let rust = Phonemizer::new().unwrap();

        for sentence in COMPARISON_CORPUS {
            let rust_out = rust_sentence_phonemes(&rust, sentence);
            let espeak_out = espeak_sentence_phonemes(&espeak, sentence);
            assert!(
                !rust_out.trim().is_empty(),
                "rust output is empty for: {sentence}"
            );
            assert!(
                !espeak_out.trim().is_empty(),
                "espeak output is empty for: {sentence}"
            );
            assert_eq!(
                rust_out, espeak_out,
                "raw corpus mismatch for input={sentence:?}\nrust={rust_out:?}\nespeak={espeak_out:?}"
            );
        }
    }
}
