//! This module provides functionality for interacting with the [OpenAI Models API](https://platform.openai.com/docs/api-reference/models).
//!
//! The Models API allows you to retrieve a list of available models and fetch details about a specific model.
//! These endpoints are useful for determining which model IDs you can use for chat, embeddings, moderation,
//! image/audio workloads, and legacy endpoints.
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
/// Note that some fields-like `permission`, `root`, and `parent`-might not be returned by
/// all model objects. They are made optional (or defaulted) to avoid serde decoding errors.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// The unique identifier for the model, e.g. `"gpt-4o-mini"`.
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
    #[serde(rename = "object")]
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

macro_rules! define_models {
    ($($variant:ident => $id:literal,)+) => {
        /// Known OpenAI model identifiers.
        ///
        /// This enum is generated from a docs-curated list of model aliases and dated snapshots.
        /// Unknown model IDs always deserialize as [`Model::Other`].
        #[derive(Debug, Clone, PartialEq)]
        #[non_exhaustive]
        pub enum Model {
            $(
                /// A known OpenAI model identifier.
                $variant,
            )+
            /// A catch-all for unknown or future model names.
            Other(String),
        }

        impl Default for Model {
            fn default() -> Self {
                Self::Gpt4oMini
            }
        }

        fn parse_model_str(s: &str) -> Model {
            match s {
                $(
                    $id => Model::$variant,
                )+
                _ => Model::Other(s.to_owned()),
            }
        }

        impl Model {
            /// Returns the string identifier associated with this [`Model`] variant.
            pub fn as_str(&self) -> &str {
                match self {
                    $(
                        Model::$variant => $id,
                    )+
                    Model::Other(s) => s.as_str(),
                }
            }

            #[cfg(test)]
            fn known_variants() -> Vec<Model> {
                vec![
                    $(
                        Model::$variant,
                    )+
                ]
            }
        }
    };
}

define_models! {
    Babbage002 => "babbage-002",
    ChatGpt4oLatest => "chatgpt-4o-latest",
    ChatGptImageLatest => "chatgpt-image-latest",
    CodexMiniLatest => "codex-mini-latest",
    ComputerUsePreview => "computer-use-preview",
    DallE2 => "dall-e-2",
    DallE3 => "dall-e-3",
    Davinci002 => "davinci-002",
    Gpt3_5Turbo => "gpt-3.5-turbo",
    Gpt4 => "gpt-4",
    Gpt4Turbo => "gpt-4-turbo",
    Gpt4TurboPreview => "gpt-4-turbo-preview",
    Gpt41 => "gpt-4.1",
    Gpt41Mini => "gpt-4.1-mini",
    Gpt41Nano => "gpt-4.1-nano",
    Gpt45Preview => "gpt-4.5-preview",
    Gpt4o => "gpt-4o",
    Gpt4oAudioPreview => "gpt-4o-audio-preview",
    Gpt4oMini => "gpt-4o-mini",
    Gpt4oMiniAudioPreview => "gpt-4o-mini-audio-preview",
    Gpt4oMiniRealtimePreview => "gpt-4o-mini-realtime-preview",
    Gpt4oMiniSearchPreview => "gpt-4o-mini-search-preview",
    Gpt4oMiniTranscribe => "gpt-4o-mini-transcribe",
    Gpt4oMiniTts => "gpt-4o-mini-tts",
    Gpt4oRealtimePreview => "gpt-4o-realtime-preview",
    Gpt4oSearchPreview => "gpt-4o-search-preview",
    Gpt4oTranscribe => "gpt-4o-transcribe",
    Gpt4oTranscribeDiarize => "gpt-4o-transcribe-diarize",
    Gpt5 => "gpt-5",
    Gpt5ChatLatest => "gpt-5-chat-latest",
    Gpt5Codex => "gpt-5-codex",
    Gpt5Mini => "gpt-5-mini",
    Gpt5Nano => "gpt-5-nano",
    Gpt5Pro => "gpt-5-pro",
    Gpt51 => "gpt-5.1",
    Gpt51ChatLatest => "gpt-5.1-chat-latest",
    Gpt51Codex => "gpt-5.1-codex",
    Gpt51CodexMax => "gpt-5.1-codex-max",
    Gpt51CodexMini => "gpt-5.1-codex-mini",
    Gpt52 => "gpt-5.2",
    Gpt52ChatLatest => "gpt-5.2-chat-latest",
    Gpt52Codex => "gpt-5.2-codex",
    Gpt52Pro => "gpt-5.2-pro",
    GptAudio => "gpt-audio",
    GptAudioMini => "gpt-audio-mini",
    GptImage1 => "gpt-image-1",
    GptImage1Mini => "gpt-image-1-mini",
    GptImage15 => "gpt-image-1.5",
    GptOss120b => "gpt-oss-120b",
    GptOss20b => "gpt-oss-20b",
    GptRealtime => "gpt-realtime",
    GptRealtimeMini => "gpt-realtime-mini",
    O1 => "o1",
    O1Mini => "o1-mini",
    O1Preview => "o1-preview",
    O1Pro => "o1-pro",
    O3 => "o3",
    O3DeepResearch => "o3-deep-research",
    O3Mini => "o3-mini",
    O3Pro => "o3-pro",
    O4Mini => "o4-mini",
    O4MiniDeepResearch => "o4-mini-deep-research",
    OmniModeration => "omni-moderation",
    OmniModerationLatest => "omni-moderation-latest",
    Sora2 => "sora-2",
    Sora2Pro => "sora-2-pro",
    TextEmbedding3Large => "text-embedding-3-large",
    TextEmbedding3Small => "text-embedding-3-small",
    TextEmbeddingAda002 => "text-embedding-ada-002",
    TextModerationLatest => "text-moderation-latest",
    TextModerationStable => "text-moderation-stable",
    Tts1 => "tts-1",
    Tts1Hd => "tts-1-hd",
    Whisper1 => "whisper-1",
    ComputerUsePreview2025_03_11 => "computer-use-preview-2025-03-11",
    Gpt4Turbo2024_04_09 => "gpt-4-turbo-2024-04-09",
    Gpt41_2025_04_14 => "gpt-4.1-2025-04-14",
    Gpt41Mini2025_04_14 => "gpt-4.1-mini-2025-04-14",
    Gpt41Nano2025_04_14 => "gpt-4.1-nano-2025-04-14",
    Gpt45Preview2025_02_27 => "gpt-4.5-preview-2025-02-27",
    Gpt4o2024_05_13 => "gpt-4o-2024-05-13",
    Gpt4o2024_08_06 => "gpt-4o-2024-08-06",
    Gpt4o2024_11_20 => "gpt-4o-2024-11-20",
    Gpt4oAudioPreview2024_10_01 => "gpt-4o-audio-preview-2024-10-01",
    Gpt4oAudioPreview2024_12_17 => "gpt-4o-audio-preview-2024-12-17",
    Gpt4oAudioPreview2025_06_03 => "gpt-4o-audio-preview-2025-06-03",
    Gpt4oMini2024_07_18 => "gpt-4o-mini-2024-07-18",
    Gpt4oMiniAudioPreview2024_12_17 => "gpt-4o-mini-audio-preview-2024-12-17",
    Gpt4oMiniRealtimePreview2024_12_17 => "gpt-4o-mini-realtime-preview-2024-12-17",
    Gpt4oMiniSearchPreview2025_03_11 => "gpt-4o-mini-search-preview-2025-03-11",
    Gpt4oMiniTranscribe2025_03_20 => "gpt-4o-mini-transcribe-2025-03-20",
    Gpt4oMiniTranscribe2025_12_15 => "gpt-4o-mini-transcribe-2025-12-15",
    Gpt4oMiniTts2025_03_20 => "gpt-4o-mini-tts-2025-03-20",
    Gpt4oMiniTts2025_12_15 => "gpt-4o-mini-tts-2025-12-15",
    Gpt4oRealtimePreview2024_10_01 => "gpt-4o-realtime-preview-2024-10-01",
    Gpt4oRealtimePreview2024_12_17 => "gpt-4o-realtime-preview-2024-12-17",
    Gpt4oRealtimePreview2025_06_03 => "gpt-4o-realtime-preview-2025-06-03",
    Gpt4oSearchPreview2025_03_11 => "gpt-4o-search-preview-2025-03-11",
    Gpt5_2025_08_07 => "gpt-5-2025-08-07",
    Gpt5Mini2025_08_07 => "gpt-5-mini-2025-08-07",
    Gpt5Nano2025_08_07 => "gpt-5-nano-2025-08-07",
    Gpt5Pro2025_10_06 => "gpt-5-pro-2025-10-06",
    Gpt51_2025_11_13 => "gpt-5.1-2025-11-13",
    Gpt52_2025_12_11 => "gpt-5.2-2025-12-11",
    Gpt52Pro2025_12_11 => "gpt-5.2-pro-2025-12-11",
    GptAudio2025_08_28 => "gpt-audio-2025-08-28",
    GptAudioMini2025_10_06 => "gpt-audio-mini-2025-10-06",
    GptAudioMini2025_12_15 => "gpt-audio-mini-2025-12-15",
    GptImage15_2025_12_16 => "gpt-image-1.5-2025-12-16",
    GptRealtime2025_08_28 => "gpt-realtime-2025-08-28",
    GptRealtimeMini2025_10_06 => "gpt-realtime-mini-2025-10-06",
    GptRealtimeMini2025_12_15 => "gpt-realtime-mini-2025-12-15",
    O12024_12_17 => "o1-2024-12-17",
    O1Mini2024_09_12 => "o1-mini-2024-09-12",
    O1Preview2024_09_12 => "o1-preview-2024-09-12",
    O1Pro2025_03_19 => "o1-pro-2025-03-19",
    O3_2025_04_16 => "o3-2025-04-16",
    O3DeepResearch2025_06_26 => "o3-deep-research-2025-06-26",
    O3Mini2025_01_31 => "o3-mini-2025-01-31",
    O3Pro2025_06_10 => "o3-pro-2025-06-10",
    O4Mini2025_04_16 => "o4-mini-2025-04-16",
    O4MiniDeepResearch2025_06_26 => "o4-mini-deep-research-2025-06-26",
    OmniModeration2024_09_26 => "omni-moderation-2024-09-26",
    Sora2_2025_10_06 => "sora-2-2025-10-06",
    Sora2_2025_12_08 => "sora-2-2025-12-08",
    Sora2Pro2025_10_06 => "sora-2-pro-2025-10-06",
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
    /// 1. **list_models** - success & error
    /// 2. **retrieve_model** - success & error
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_list_models_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o-mini",
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
                    "root": "gpt-4o-mini",
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

        let result = list_models(&client).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let models = result.unwrap();
        assert_eq!(models.len(), 1);
        let first_model = &models[0];
        assert_eq!(first_model.id, "gpt-4o-mini");
        assert_eq!(first_model.object, "model");
        assert_eq!(first_model.owned_by, "openai");
        assert!(!first_model.permission.is_empty());
        assert_eq!(first_model.root.as_deref(), Some("gpt-4o-mini"));
    }

    #[tokio::test]
    async fn test_list_models_api_error() {
        let mock_server = MockServer::start().await;

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
            "id": "gpt-4.1-mini",
            "object": "model",
            "created": 1673645000,
            "owned_by": "openai",
            "permission": [],
            "root": "gpt-4.1-mini",
            "parent": null
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/models/gpt-4\.1-mini$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_model(&client, "gpt-4.1-mini").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let model = result.unwrap();
        assert_eq!(model.id, "gpt-4.1-mini");
        assert_eq!(model.object, "model");
        assert_eq!(model.owned_by, "openai");
        assert_eq!(model.permission.len(), 0);
        assert_eq!(model.root.as_deref(), Some("gpt-4.1-mini"));
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

    #[test]
    fn test_model_round_trip() {
        for variant in Model::known_variants() {
            let string_id = variant.as_str();
            let parsed_variant = Model::from(string_id);
            assert_eq!(
                variant, parsed_variant,
                "Expected round trip to yield {variant:?}, but got {parsed_variant:?}",
            );
        }

        let unknown_str = "my-awesome-custom-model";
        let other_variant = Model::Other(unknown_str.to_string());
        let from_str = Model::from(other_variant.as_str());
        assert_eq!(
            other_variant, from_str,
            "Custom model string did not properly round-trip."
        );
    }
}
