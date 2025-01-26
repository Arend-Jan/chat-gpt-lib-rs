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
use serde::Deserialize;

/// Represents an OpenAI model.
///
/// Note that some fields—like `permission`, `root`, and `parent`—might not be returned by
/// all model objects. They are made optional (or defaulted) to avoid serde decoding errors.
#[derive(Debug, Deserialize)]
pub struct Model {
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
/// Each [`Model`] includes a vector of these to indicate usage rights, rate limit policies, etc.
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
/// containing a list of [`Model`] objects.
#[derive(Debug, Deserialize)]
struct ModelList {
    /// Indicates the object type (usually `"list"`).
    #[serde(rename = "object")] // keep the object for potential later use
    _object: String,
    /// The actual list of models.
    data: Vec<Model>,
}

/// Retrieves a list of all available models from the OpenAI API.
///
/// Corresponds to `GET /v1/models`.
///
/// # Errors
///
/// Returns an [`OpenAIError`] if the network request fails or the response
/// cannot be parsed.
pub async fn list_models(client: &OpenAIClient) -> Result<Vec<Model>, OpenAIError> {
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
pub async fn retrieve_model(client: &OpenAIClient, model_id: &str) -> Result<Model, OpenAIError> {
    let endpoint = format!("models/{}", model_id);
    get_json(client, &endpoint).await
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
