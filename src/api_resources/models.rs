//! This module provides functionality for interacting with the [OpenAI Models API](https://platform.openai.com/docs/api-reference/models).
//!
//! The Models API allows you to retrieve a list of available models and fetch details about a specific model.
//! These endpoints are useful for determining which model IDs you can use for completions, chat, embeddings, and other functionalities.
//!
//! # Usage
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::models;
//! use chat_gpt_lib_rs::{OpenAIClient, OpenAIError};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?;
//!
//!     // List all available models.
//!     let all_models = models::list_models(&client).await?;
//!     println!("Available Models: {:?}", all_models);
//!
//!     // Retrieve a specific model by ID.
//!     if let Some(first_model) = all_models.first() {
//!         let model_id = &first_model.id;
//!         let retrieved = models::retrieve_model(&client, model_id).await?;
//!         println!("Retrieved Model: {:?}", retrieved);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::api::get_json;
use crate::config::OpenAIClient;
use crate::error::OpenAIError;
use serde::{Deserialize, Serialize};

/// Represents an OpenAI model (detailed info from API).
///
/// Note that some fields—like `permission`, `root`, and `parent`—might not be returned by
/// all model objects. They are made optional (or defaulted) to avoid serde decoding errors.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// The unique identifier for the model, e.g. `"text-davinci-003"`.
    pub id: String,
    /// The object type, usually `"model"`.
    pub object: String,
    /// Unix timestamp (in seconds) when this model was created, if available.
    #[serde(default)]
    pub created: Option<u64>,
    /// The owner of this model, often `"openai"` or an organization ID.
    pub owned_by: String,
    /// Some endpoints return a list of permission objects; others may omit it.
    #[serde(default)]
    pub permission: Vec<ModelPermission>,
    /// For certain models, a `"root"` field may indicate the parent or root model.
    #[serde(default)]
    pub root: Option<String>,
    /// For certain models, a `"parent"` field references the parent model.
    #[serde(default)]
    pub parent: Option<String>,
}

/// Describes permissions for a model.
///
/// Each [`ModelInfo`] includes a vector of these to indicate usage rights, rate limit policies, etc.
/// This may be absent in some model responses.
#[derive(Debug, Deserialize)]
pub struct ModelPermission {
    /// The unique identifier for the permission entry.
    pub id: String,
    /// The object type, usually `"model_permission"`.
    pub object: String,
    /// Unix timestamp (in seconds) when this permission was created.
    pub created: u64,
    /// Indicates if the user can create fine-tuned engines from this model.
    pub allow_create_engine: bool,
    /// Indicates if the user is permitted to sample from the model.
    pub allow_sampling: bool,
    /// Indicates if the user can request log probabilities.
    pub allow_logprobs: bool,
    /// Indicates if the user can access search indices.
    pub allow_search_indices: bool,
    /// Indicates if the user can view the model details.
    pub allow_view: bool,
    /// Indicates if the user can fine-tune the model.
    pub allow_fine_tuning: bool,
    /// The organization this permission applies to.
    pub organization: String,
    /// Group identifier, if applicable.
    #[serde(default)]
    pub group: Option<String>,
    /// Whether this permission is blocking some other action or usage.
    pub is_blocking: bool,
}

/// A wrapper for the response returned by the `GET /v1/models` endpoint,
/// containing a list of [`ModelInfo`] objects.
#[derive(Debug, Deserialize)]
struct ModelList {
    /// Indicates the object type (usually `"list"`).
    #[serde(rename = "object")] // keep the object for potential later use
    _object: String,
    /// The actual list of models.
    data: Vec<ModelInfo>,
}

/// Retrieves a list of all available models from the OpenAI API.
///
/// Corresponds to `GET /v1/models`.
///
/// # Errors
///
/// Returns an [`OpenAIError`] if the network request fails or the response
/// cannot be parsed.
pub async fn list_models(client: &OpenAIClient) -> Result<Vec<ModelInfo>, OpenAIError> {
    let endpoint = "models";
    let response: ModelList = get_json(client, endpoint).await?;
    Ok(response.data)
}

/// Retrieves the details for a specific model by its `model_id`.
///
/// Corresponds to `GET /v1/models/{model_id}`.
///
/// # Errors
///
/// Returns an [`OpenAIError`] if the network request fails, the response
/// cannot be parsed, or the model does not exist.
pub async fn retrieve_model(
    client: &OpenAIClient,
    model_id: &str,
) -> Result<ModelInfo, OpenAIError> {
    let endpoint = format!("models/{}", model_id);
    get_json(client, &endpoint).await
}

/// An enum representing known OpenAI model identifiers, plus an `Other` variant for unrecognized or custom model IDs.
///
/// - **`Gpt4oMiniAudioPreview`** => `"gpt-4o-mini-audio-preview"`  
/// - **`DallE2`** => `"dall-e-2"`  
/// - etc.
///
/// We manually implement `Serialize`/`Deserialize` so that known variants map exactly to the
/// expected strings, and all unknown strings map to `Other(String)`.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum Model {
    /// The `gpt-4.5-preview` model (owned by system)
    Gpt45Preview,
    /// The `gpt-4.5-preview-2025-02-27` model (owned by system)
    Gpt45Preview2025_02_27,
    /// The `gpt-4o-mini-audio-preview` model (owned by system).
    Gpt4oMiniAudioPreview,
    /// The `gpt-4o-mini-audio-preview-2024-12-17` model (owned by system).
    Gpt4oMiniAudioPreview2024_12_17,
    /// The `gpt-4o-mini-realtime-preview` model (owned by system).
    Gpt4oMiniRealtimePreview,
    /// The `dall-e-2` model (owned by system).
    DallE2,
    /// The `gpt-4o-2024-11-20` model (owned by system).
    Gpt4o2024_11_20,
    /// The `o1-mini-2024-09-12` model (owned by system).
    O1Mini2024_09_12,
    /// The `o1-preview-2024-09-12` model (owned by system).
    O1Preview2024_09_12,
    /// The `o1-mini` model (owned by system).
    #[default]
    O1Mini,
    /// The `o1-preview` model (owned by system).
    O1Preview,
    /// The `chatgpt-4o-latest` model (owned by system).
    ChatGpt4oLatest,
    /// The `whisper-1` model (owned by openai-internal).
    Whisper1,
    /// The `dall-e-3` model (owned by system).
    DallE3,
    /// The `gpt-4-turbo` model (owned by system).
    Gpt4Turbo,
    /// The `gpt-4-turbo-preview` model (owned by system).
    Gpt4TurboPreview,
    /// The `gpt-4o-audio-preview` model (owned by system).
    Gpt4oAudioPreview,
    /// The `gpt-4o-audio-preview-2024-10-01` model (owned by system).
    Gpt4oAudioPreview2024_10_01,
    /// The `babbage-002` model (owned by system).
    Babbage002,
    /// The `omni-moderation-latest` model (owned by system).
    OmniModerationLatest,
    /// The `omni-moderation-2024-09-26` model (owned by system).
    OmniModeration2024_09_26,
    /// The `tts-1-hd-1106` model (owned by system).
    Tts1Hd1106,
    /// The `gpt-4o-2024-08-06` model (owned by system).
    Gpt4o2024_08_06,
    /// The `gpt-4o` model (owned by system).
    Gpt4o,
    /// The `gpt-4o-2024-05-13` model (owned by system).
    Gpt4o2024_05_13,
    /// The `tts-1-hd` model (owned by system).
    Tts1Hd,
    /// The `gpt-4-turbo-2024-04-09` model (owned by system).
    Gpt4Turbo2024_04_09,
    /// The `tts-1` model (owned by openai-internal).
    Tts1,
    /// The `gpt-3.5-turbo-16k` model (owned by openai-internal).
    Gpt3_5Turbo16k,
    /// The `tts-1-1106` model (owned by system).
    Tts1_1106,
    /// The `davinci-002` model (owned by system).
    Davinci002,
    /// The `gpt-3.5-turbo-1106` model (owned by system).
    Gpt3_5Turbo1106,
    /// The `gpt-4o-mini-realtime-preview-2024-12-17` model (owned by system).
    Gpt4oMiniRealtimePreview2024_12_17,
    /// The `gpt-3.5-turbo-instruct` model (owned by system).
    Gpt3_5TurboInstruct,
    /// The `gpt-4o-realtime-preview-2024-10-01` model (owned by system).
    Gpt4oRealtimePreview2024_10_01,
    /// The `gpt-3.5-turbo-instruct-0914` model (owned by system).
    Gpt3_5TurboInstruct0914,
    /// The `gpt-3.5-turbo-0125` model (owned by system).
    Gpt3_5Turbo0125,
    /// The `gpt-4o-audio-preview-2024-12-17` model (owned by system).
    Gpt4oAudioPreview2024_12_17,
    /// The `gpt-4o-realtime-preview-2024-12-17` model (owned by system).
    Gpt4oRealtimePreview2024_12_17,
    /// The `gpt-3.5-turbo` model (owned by openai).
    Gpt3_5Turbo,
    /// The `text-embedding-3-large` model (owned by system).
    TextEmbedding3Large,
    /// The `gpt-4o-realtime-preview` model (owned by system).
    Gpt4oRealtimePreview,
    /// The `text-embedding-3-small` model (owned by system).
    TextEmbedding3Small,
    /// The `gpt-4-0125-preview` model (owned by system).
    Gpt40125Preview,
    /// The `gpt-4` model (owned by openai).
    Gpt4,
    /// The `text-embedding-ada-002` model (owned by openai-internal).
    TextEmbeddingAda002,
    /// The `gpt-4-1106-preview` model (owned by system).
    Gpt40106Preview,
    /// The `gpt-4o-mini` model (owned by system).
    Gpt4oMini,
    /// The `gpt-4-0613` model (owned by openai).
    Gpt40613,
    /// The `gpt-4o-mini-2024-07-18` model (owned by system).
    Gpt4oMini2024_07_18,
    /// The `gpt-4.1-nano` model (owned by system).
    Gpt41Nano,
    /// The `gpt-4.1-nano-2025-04-14` model (owned by system).
    Gpt41Nano2025_04_14,
    /// The `gpt-4.1-mini` model (owned by system).
    Gpt41Mini,
    /// The `gpt-4.1-mini-2025-04-14` model (owned by system).
    Gpt41Mini2025_04_14,
    /// The `gpt-4.1` model (owned by system).
    Gpt41,
    /// The `gpt-4.1-2025-04-14` model (owned by system).
    Gpt41_2025_04_14,
    /// The `gpt-4o-mini-search-preview` model (owned by system).
    Gpt4oMiniSearchPreview,
    /// The `gpt-4o-mini-search-preview-2025-03-11` model (owned by system).
    Gpt4oMiniSearchPreview2025_03_11,
    /// The `gpt-4o-search-preview` model (owned by system).
    Gpt4oSearchPreview,
    /// The `gpt-4o-search-preview-2025-03-11` model (owned by system).
    Gpt4oSearchPreview2025_03_11,
    /// The `gpt-4o-mini-tts` model (owned by system).
    Gpt4oMiniTts,
    /// The `gpt-4o-mini-transcribe` model (owned by system).
    Gpt4oMiniTranscribe,
    /// The `gpt-4o-transcribe` model (owned by system).
    Gpt4oTranscribe,
    /// The `gpt-image-1` model (owned by system).
    GptImage1,
    /// The `o1-2024-12-17` model (owned by system).
    O12024_12_17,
    /// The `o1` model (owned by system).
    O1,
    /// The `o1-pro` model (owned by system).
    O1Pro,
    /// The `o1-pro-2025-03-19` model (owned by system).
    O1Pro2025_03_19,
    /// The `o3-mini` model (owned by system).
    O3Mini,
    /// The `o3-mini-2025-01-31` model (owned by system).
    O3Mini2025_01_31,
    /// The `o4-mini` model (owned by system).
    O4Mini,
    /// The `o4-mini-2025-04-16` model (owned by system).
    O4Mini2025_04_16,

    /// A catch-all for unknown or future model names.
    Other(String),
}

/// Internal helper to convert a model string into a known variant, or `Other(...)` if unrecognized.
fn parse_model_str(s: &str) -> Model {
    match s {
        "gpt-4.5-preview" => Model::Gpt45Preview,
        "gpt-4.5-preview-2025-02-27" => Model::Gpt45Preview2025_02_27,
        "gpt-4o-mini-audio-preview" => Model::Gpt4oMiniAudioPreview,
        "gpt-4o-mini-audio-preview-2024-12-17" => Model::Gpt4oMiniAudioPreview2024_12_17,
        "gpt-4o-mini-realtime-preview" => Model::Gpt4oMiniRealtimePreview,
        "dall-e-2" => Model::DallE2,
        "gpt-4o-2024-11-20" => Model::Gpt4o2024_11_20,
        "o1-mini-2024-09-12" => Model::O1Mini2024_09_12,
        "o1-preview-2024-09-12" => Model::O1Preview2024_09_12,
        "o1-mini" => Model::O1Mini,
        "o1-preview" => Model::O1Preview,
        "chatgpt-4o-latest" => Model::ChatGpt4oLatest,
        "whisper-1" => Model::Whisper1,
        "dall-e-3" => Model::DallE3,
        "gpt-4-turbo" => Model::Gpt4Turbo,
        "gpt-4-turbo-preview" => Model::Gpt4TurboPreview,
        "gpt-4o-audio-preview" => Model::Gpt4oAudioPreview,
        "gpt-4o-audio-preview-2024-10-01" => Model::Gpt4oAudioPreview2024_10_01,
        "babbage-002" => Model::Babbage002,
        "omni-moderation-latest" => Model::OmniModerationLatest,
        "omni-moderation-2024-09-26" => Model::OmniModeration2024_09_26,
        "tts-1-hd-1106" => Model::Tts1Hd1106,
        "gpt-4o-2024-08-06" => Model::Gpt4o2024_08_06,
        "gpt-4o" => Model::Gpt4o,
        "gpt-4o-2024-05-13" => Model::Gpt4o2024_05_13,
        "tts-1-hd" => Model::Tts1Hd,
        "gpt-4-turbo-2024-04-09" => Model::Gpt4Turbo2024_04_09,
        "tts-1" => Model::Tts1,
        "gpt-3.5-turbo-16k" => Model::Gpt3_5Turbo16k,
        "tts-1-1106" => Model::Tts1_1106,
        "davinci-002" => Model::Davinci002,
        "gpt-3.5-turbo-1106" => Model::Gpt3_5Turbo1106,
        "gpt-4o-mini-realtime-preview-2024-12-17" => Model::Gpt4oMiniRealtimePreview2024_12_17,
        "gpt-3.5-turbo-instruct" => Model::Gpt3_5TurboInstruct,
        "gpt-4o-realtime-preview-2024-10-01" => Model::Gpt4oRealtimePreview2024_10_01,
        "gpt-3.5-turbo-instruct-0914" => Model::Gpt3_5TurboInstruct0914,
        "gpt-3.5-turbo-0125" => Model::Gpt3_5Turbo0125,
        "gpt-4o-audio-preview-2024-12-17" => Model::Gpt4oAudioPreview2024_12_17,
        "gpt-4o-realtime-preview-2024-12-17" => Model::Gpt4oRealtimePreview2024_12_17,
        "gpt-3.5-turbo" => Model::Gpt3_5Turbo,
        "text-embedding-3-large" => Model::TextEmbedding3Large,
        "gpt-4o-realtime-preview" => Model::Gpt4oRealtimePreview,
        "text-embedding-3-small" => Model::TextEmbedding3Small,
        "gpt-4-0125-preview" => Model::Gpt40125Preview,
        "gpt-4" => Model::Gpt4,
        "text-embedding-ada-002" => Model::TextEmbeddingAda002,
        "gpt-4-1106-preview" => Model::Gpt40106Preview,
        "gpt-4o-mini" => Model::Gpt4oMini,
        "gpt-4-0613" => Model::Gpt40613,
        "gpt-4o-mini-2024-07-18" => Model::Gpt4oMini2024_07_18,
        "gpt-4.1-nano" => Model::Gpt41Nano,
        "gpt-4.1-nano-2025-04-14" => Model::Gpt41Nano2025_04_14,
        "gpt-4.1-mini" => Model::Gpt41Mini,
        "gpt-4.1-mini-2025-04-14" => Model::Gpt41Mini2025_04_14,
        "gpt-4.1" => Model::Gpt41,
        "gpt-4.1-2025-04-14" => Model::Gpt41_2025_04_14,
        "gpt-4o-mini-search-preview" => Model::Gpt4oMiniSearchPreview,
        "gpt-4o-mini-search-preview-2025-03-11" => Model::Gpt4oMiniSearchPreview2025_03_11,
        "gpt-4o-search-preview" => Model::Gpt4oSearchPreview,
        "gpt-4o-search-preview-2025-03-11" => Model::Gpt4oSearchPreview2025_03_11,
        "gpt-4o-mini-tts" => Model::Gpt4oMiniTts,
        "gpt-4o-mini-transcribe" => Model::Gpt4oMiniTranscribe,
        "gpt-4o-transcribe" => Model::Gpt4oTranscribe,
        "gpt-image-1" => Model::GptImage1,
        "o1-2024-12-17" => Model::O12024_12_17,
        "o1" => Model::O1,
        "o1-pro" => Model::O1Pro,
        "o1-pro-2025-03-19" => Model::O1Pro2025_03_19,
        "o3-mini" => Model::O3Mini,
        "o3-mini-2025-01-31" => Model::O3Mini2025_01_31,
        "o4-mini" => Model::O4Mini,
        "o4-mini-2025-04-16" => Model::O4Mini2025_04_16,
        _ => Model::Other(s.to_owned()),
    }
}

impl From<&str> for Model {
    fn from(s: &str) -> Self {
        parse_model_str(s)
    }
}

impl From<String> for Model {
    fn from(s: String) -> Self {
        parse_model_str(&s)
    }
}

impl From<&String> for Model {
    fn from(s: &String) -> Self {
        parse_model_str(s)
    }
}

impl Model {
    /// Returns the string identifier (e.g. `"gpt-4o-mini-audio-preview"`) associated
    /// with this [`Model`] variant.
    pub fn as_str(&self) -> &str {
        match self {
            Model::Gpt45Preview => "gpt-4.5-preview",
            Model::Gpt45Preview2025_02_27 => "gpt-4.5-preview-2025-02-27",
            Model::Gpt4oMiniAudioPreview => "gpt-4o-mini-audio-preview",
            Model::Gpt4oMiniAudioPreview2024_12_17 => "gpt-4o-mini-audio-preview-2024-12-17",
            Model::Gpt4oMiniRealtimePreview => "gpt-4o-mini-realtime-preview",
            Model::DallE2 => "dall-e-2",
            Model::Gpt4o2024_11_20 => "gpt-4o-2024-11-20",
            Model::O1Mini2024_09_12 => "o1-mini-2024-09-12",
            Model::O1Preview2024_09_12 => "o1-preview-2024-09-12",
            Model::O1Mini => "o1-mini",
            Model::O1Preview => "o1-preview",
            Model::ChatGpt4oLatest => "chatgpt-4o-latest",
            Model::Whisper1 => "whisper-1",
            Model::DallE3 => "dall-e-3",
            Model::Gpt4Turbo => "gpt-4-turbo",
            Model::Gpt4TurboPreview => "gpt-4-turbo-preview",
            Model::Gpt4oAudioPreview => "gpt-4o-audio-preview",
            Model::Gpt4oAudioPreview2024_10_01 => "gpt-4o-audio-preview-2024-10-01",
            Model::Babbage002 => "babbage-002",
            Model::OmniModerationLatest => "omni-moderation-latest",
            Model::OmniModeration2024_09_26 => "omni-moderation-2024-09-26",
            Model::Tts1Hd1106 => "tts-1-hd-1106",
            Model::Gpt4o2024_08_06 => "gpt-4o-2024-08-06",
            Model::Gpt4o => "gpt-4o",
            Model::Gpt4o2024_05_13 => "gpt-4o-2024-05-13",
            Model::Tts1Hd => "tts-1-hd",
            Model::Gpt4Turbo2024_04_09 => "gpt-4-turbo-2024-04-09",
            Model::Tts1 => "tts-1",
            Model::Gpt3_5Turbo16k => "gpt-3.5-turbo-16k",
            Model::Tts1_1106 => "tts-1-1106",
            Model::Davinci002 => "davinci-002",
            Model::Gpt3_5Turbo1106 => "gpt-3.5-turbo-1106",
            Model::Gpt4oMiniRealtimePreview2024_12_17 => "gpt-4o-mini-realtime-preview-2024-12-17",
            Model::Gpt3_5TurboInstruct => "gpt-3.5-turbo-instruct",
            Model::Gpt4oRealtimePreview2024_10_01 => "gpt-4o-realtime-preview-2024-10-01",
            Model::Gpt3_5TurboInstruct0914 => "gpt-3.5-turbo-instruct-0914",
            Model::Gpt3_5Turbo0125 => "gpt-3.5-turbo-0125",
            Model::Gpt4oAudioPreview2024_12_17 => "gpt-4o-audio-preview-2024-12-17",
            Model::Gpt4oRealtimePreview2024_12_17 => "gpt-4o-realtime-preview-2024-12-17",
            Model::Gpt3_5Turbo => "gpt-3.5-turbo",
            Model::TextEmbedding3Large => "text-embedding-3-large",
            Model::Gpt4oRealtimePreview => "gpt-4o-realtime-preview",
            Model::TextEmbedding3Small => "text-embedding-3-small",
            Model::Gpt40125Preview => "gpt-4-0125-preview",
            Model::Gpt4 => "gpt-4",
            Model::TextEmbeddingAda002 => "text-embedding-ada-002",
            Model::Gpt40106Preview => "gpt-4-1106-preview",
            Model::Gpt4oMini => "gpt-4o-mini",
            Model::Gpt40613 => "gpt-4-0613",
            Model::Gpt4oMini2024_07_18 => "gpt-4o-mini-2024-07-18",
            Model::Gpt41Nano => "gpt-4.1-nano",
            Model::Gpt41Nano2025_04_14 => "gpt-4.1-nano-2025-04-14",
            Model::Gpt41Mini => "gpt-4.1-mini",
            Model::Gpt41Mini2025_04_14 => "gpt-4.1-mini-2025-04-14",
            Model::Gpt41 => "gpt-4.1",
            Model::Gpt41_2025_04_14 => "gpt-4.1-2025-04-14",
            Model::Gpt4oMiniSearchPreview => "gpt-4o-mini-search-preview",
            Model::Gpt4oMiniSearchPreview2025_03_11 => "gpt-4o-mini-search-preview-2025-03-11",
            Model::Gpt4oSearchPreview => "gpt-4o-search-preview",
            Model::Gpt4oSearchPreview2025_03_11 => "gpt-4o-search-preview-2025-03-11",
            Model::Gpt4oMiniTts => "gpt-4o-mini-tts",
            Model::Gpt4oMiniTranscribe => "gpt-4o-mini-transcribe",
            Model::Gpt4oTranscribe => "gpt-4o-transcribe",
            Model::GptImage1 => "gpt-image-1",
            Model::O12024_12_17 => "o1-2024-12-17",
            Model::O1 => "o1",
            Model::O1Pro => "o1-pro",
            Model::O1Pro2025_03_19 => "o1-pro-2025-03-19",
            Model::O3Mini => "o3-mini",
            Model::O3Mini2025_01_31 => "o3-mini-2025-01-31",
            Model::O4Mini => "o4-mini",
            Model::O4Mini2025_04_16 => "o4-mini-2025-04-16",
            Model::Other(s) => s.as_str(),
        }
    }
}

impl Serialize for Model {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D>(deserializer: D) -> Result<Model, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(parse_model_str(&s))
    }
}

#[cfg(test)]
mod tests {
    /// # Tests for the `models` module
    ///
    /// We use [`wiremock`](https://crates.io/crates/wiremock) to simulate responses from
    /// the **Models** API (`GET /v1/models` and `GET /v1/models/{id}`).
    /// This covers:
    /// 1. **list_models** – success & error
    /// 2. **retrieve_model** – success & error
    ///
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_list_models_success() {
        // Start a local mock server
        let mock_server = MockServer::start().await;

        // Define a successful JSON response
        let success_body = json!({
            "object": "list",
            "data": [
                {
                    "id": "text-davinci-003",
                    "object": "model",
                    "created": 1673643147,
                    "owned_by": "openai",
                    "permission": [
                        {
                            "id": "modelperm-abc123",
                            "object": "model_permission",
                            "created": 1673643000,
                            "allow_create_engine": true,
                            "allow_sampling": true,
                            "allow_logprobs": true,
                            "allow_search_indices": true,
                            "allow_view": true,
                            "allow_fine_tuning": true,
                            "organization": "openai",
                            "group": null,
                            "is_blocking": false
                        }
                    ],
                    "root": "text-davinci-003",
                    "parent": null
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Call the function under test
        let result = list_models(&client).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let models = result.unwrap();
        assert_eq!(models.len(), 1);
        let first_model = &models[0];
        assert_eq!(first_model.id, "text-davinci-003");
        assert_eq!(first_model.object, "model");
        assert_eq!(first_model.owned_by, "openai");
        assert!(first_model.permission.len() > 0);
        assert_eq!(first_model.root.as_deref(), Some("text-davinci-003"));
    }

    #[tokio::test]
    async fn test_list_models_api_error() {
        let mock_server = MockServer::start().await;

        // Mock an error response
        let error_body = json!({
            "error": {
                "message": "Could not list models",
                "type": "server_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_models(&client).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Could not list models"));
            }
            other => panic!("Expected APIError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retrieve_model_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "id": "text-curie-001",
            "object": "model",
            "created": 1673645000,
            "owned_by": "openai",
            "permission": [],
            "root": "text-curie-001",
            "parent": null
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/models/text-curie-001$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_model(&client, "text-curie-001").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let model = result.unwrap();
        assert_eq!(model.id, "text-curie-001");
        assert_eq!(model.object, "model");
        assert_eq!(model.owned_by, "openai");
        assert_eq!(model.permission.len(), 0);
        assert_eq!(model.root.as_deref(), Some("text-curie-001"));
    }

    #[tokio::test]
    async fn test_retrieve_model_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/models/does-not-exist$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_model(&client, "does-not-exist").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Model not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }
}
