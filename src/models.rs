use std::{
    collections::HashMap,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use reqwest::blocking::Client;
use serde::Deserialize;
use thiserror::Error;

const HUGGINGFACE_ROOT: &str = "https://huggingface.co";

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum RemoteKittenModel {
    #[default]
    NanoInt8,
    NanoFp32,
    Micro,
    Mini,
}

impl RemoteKittenModel {
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::NanoInt8 => "KittenML/kitten-tts-nano-0.8-int8",
            Self::NanoFp32 => "KittenML/kitten-tts-nano-0.8-fp32",
            Self::Micro => "KittenML/kitten-tts-micro-0.8",
            Self::Mini => "KittenML/kitten-tts-mini-0.8",
        }
    }

    pub fn cache_key(self) -> &'static str {
        match self {
            Self::NanoInt8 => "kitten-tts-nano-0.8-int8",
            Self::NanoFp32 => "kitten-tts-nano-0.8-fp32",
            Self::Micro => "kitten-tts-micro-0.8",
            Self::Mini => "kitten-tts-mini-0.8",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadedModelAssets {
    pub model_path: PathBuf,
    pub voices_path: PathBuf,
    pub speed_priors: HashMap<String, f32>,
}

#[derive(Debug, Error)]
pub enum ModelDownloadError {
    #[error("io failure: {0}")]
    Io(String),
    #[error("http failure: {0}")]
    Http(String),
    #[error("invalid remote config: {0}")]
    Config(String),
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(default = "default_model_file")]
    model_file: String,
    #[serde(default = "default_voices_file")]
    voices: String,
    #[serde(default)]
    speed_priors: HashMap<String, f32>,
}

fn default_model_file() -> String {
    "model.onnx".to_string()
}

fn default_voices_file() -> String {
    "voices.npz".to_string()
}

pub fn ensure_model_downloaded(
    model: RemoteKittenModel,
) -> Result<DownloadedModelAssets, ModelDownloadError> {
    let model_cache_dir = cache_dir().join(model.cache_key());
    fs::create_dir_all(&model_cache_dir).map_err(|e| ModelDownloadError::Io(e.to_string()))?;

    let config_path = model_cache_dir.join("config.json");
    download_if_missing(
        &hf_resolve_url(model.repo_id(), "config.json"),
        &config_path,
    )?;

    let config_text =
        fs::read_to_string(&config_path).map_err(|e| ModelDownloadError::Io(e.to_string()))?;
    let config: ModelConfig = serde_json::from_str(&config_text)
        .map_err(|e| ModelDownloadError::Config(e.to_string()))?;

    if config.model_file.trim().is_empty() {
        return Err(ModelDownloadError::Config(
            "model_file is missing from config".to_string(),
        ));
    }
    if config.voices.trim().is_empty() {
        return Err(ModelDownloadError::Config(
            "voices is missing from config".to_string(),
        ));
    }

    let model_filename = config.model_file;
    let voices_filename = config.voices;
    let model_path = model_cache_dir.join(safe_filename(model_filename.as_str()));
    let voices_path = model_cache_dir.join(safe_filename(voices_filename.as_str()));

    download_if_missing(
        &hf_resolve_url(model.repo_id(), model_filename.as_str()),
        &model_path,
    )?;
    download_if_missing(
        &hf_resolve_url(model.repo_id(), voices_filename.as_str()),
        &voices_path,
    )?;

    Ok(DownloadedModelAssets {
        model_path,
        voices_path,
        speed_priors: config.speed_priors,
    })
}

fn cache_dir() -> PathBuf {
    if let Ok(custom) = env::var("KITTENTTS_CACHE_DIR") {
        return PathBuf::from(custom);
    }
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("kittentts-rs");
    }
    if let Ok(home) = env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("kittentts-rs");
    }
    if let Ok(user_profile) = env::var("USERPROFILE") {
        return PathBuf::from(user_profile)
            .join("AppData")
            .join("Local")
            .join("kittentts-rs");
    }
    PathBuf::from(".kittentts-cache")
}

fn safe_filename(name: &str) -> String {
    Path::new(name)
        .file_name()
        .and_then(|f| f.to_str())
        .map_or_else(|| name.to_string(), ToString::to_string)
}

fn hf_resolve_url(repo_id: &str, filename: &str) -> String {
    format!(
        "{}/{}/resolve/main/{}",
        HUGGINGFACE_ROOT,
        repo_id,
        filename.trim_start_matches('/')
    )
}

fn download_if_missing(url: &str, dst: &Path) -> Result<(), ModelDownloadError> {
    if dst.exists() {
        return Ok(());
    }

    let parent = dst
        .parent()
        .ok_or_else(|| ModelDownloadError::Io("invalid destination path".to_string()))?;
    fs::create_dir_all(parent).map_err(|e| ModelDownloadError::Io(e.to_string()))?;

    let client = Client::builder()
        .build()
        .map_err(|e| ModelDownloadError::Http(e.to_string()))?;
    let mut response = client
        .get(url)
        .send()
        .and_then(|r| r.error_for_status())
        .map_err(|e| ModelDownloadError::Http(e.to_string()))?;

    let mut file = File::create(dst).map_err(|e| ModelDownloadError::Io(e.to_string()))?;
    response
        .copy_to(&mut file)
        .map_err(|e| ModelDownloadError::Http(e.to_string()))?;
    file.flush()
        .map_err(|e| ModelDownloadError::Io(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{RemoteKittenModel, hf_resolve_url};

    #[test]
    fn resolve_urls_are_main_branch() {
        let repo = RemoteKittenModel::NanoInt8.repo_id();
        let url = hf_resolve_url(repo, "config.json");
        assert!(url.contains("/resolve/main/config.json"));
    }
}
