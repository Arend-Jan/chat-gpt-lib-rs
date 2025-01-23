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
    object: String,
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
