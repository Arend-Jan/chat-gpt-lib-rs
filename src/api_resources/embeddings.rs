//! This module provides functionality for creating embeddings using the
//! [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).
//!
//! The Embeddings API takes in text or tokenized text and returns a vector representation
//! (embedding) that can be used for tasks like similarity searches, clustering, or classification
//! in vector databases.
//!
//! # Overview
//!
//! The core usage involves calling [`create_embeddings`] with a [`CreateEmbeddingsRequest`],
//! which includes the `model` name (e.g., `"text-embedding-ada-002"`) and the input text(s).
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::embeddings::{create_embeddings, CreateEmbeddingsRequest, EmbeddingsInput};
//! use chat_gpt_lib_rs::error::OpenAIError;
//! use chat_gpt_lib_rs::OpenAIClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY
//!
//!     let request = CreateEmbeddingsRequest {
//!         model: "text-embedding-ada-002".to_string(),
//!         input: EmbeddingsInput::String("Hello world".to_string()),
//!         user: None,
//!     };
//!
//!     let response = create_embeddings(&client, &request).await?;
//!     for (i, emb) in response.data.iter().enumerate() {
//!         println!("Embedding #{}: vector size = {}", i, emb.embedding.len());
//!     }
//!     println!("Model used: {}", response.model);
//!     if let Some(usage) = &response.usage {
//!         println!("Usage => prompt_tokens: {}, total_tokens: {}",
//!             usage.prompt_tokens, usage.total_tokens);
//!     }
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::api::post_json;
use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// Represents the diverse ways the input can be supplied for embeddings:
///
/// - A single string
/// - Multiple strings
/// - A single sequence of token IDs
/// - Multiple sequences of token IDs
///
/// This is analogous to how prompt inputs can be specified in the Completions API,
/// so we mirror that flexibility here.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    /// A single string
    String(String),
    /// Multiple strings
    Strings(Vec<String>),
    /// A single sequence of token IDs
    Ints(Vec<i64>),
    /// Multiple sequences of token IDs
    MultiInts(Vec<Vec<i64>>),
}

/// A request struct for creating embeddings with the OpenAI API.
///
/// For more details, see the [API documentation](https://platform.openai.com/docs/api-reference/embeddings).
#[derive(Debug, Serialize, Clone)]
pub struct CreateEmbeddingsRequest {
    /// **Required.** The ID of the model to use.
    /// For example: `"text-embedding-ada-002"`.
    pub model: String,
    /// **Required.** The input text or tokens for which you want to generate embeddings.
    pub input: EmbeddingsInput,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// The response returned by the OpenAI Embeddings API.
///
/// Contains one or more embeddings (depending on whether multiple inputs were provided),
/// along with the model used and usage metrics.
#[derive(Debug, Deserialize)]
pub struct CreateEmbeddingsResponse {
    /// An identifier for this embedding request (e.g. "embedding-xxxxxx").
    pub object: String,
    /// The list of embeddings returned, each containing an index and the embedding vector.
    pub data: Vec<EmbeddingData>,
    /// The model used for creating these embeddings.
    pub model: String,
    /// Optional usage statistics for this request.
    #[serde(default)]
    pub usage: Option<EmbeddingsUsage>,
}

/// The embedding result for a single input item.
#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    /// The type of object returned, usually "embedding".
    pub object: String,
    /// The position/index of this embedding in the input array.
    pub index: u32,
    /// The embedding vector itself.
    pub embedding: Vec<f32>,
}

/// Usage statistics for an embeddings request, if provided by the API.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsUsage {
    /// Number of tokens present in the prompt(s).
    pub prompt_tokens: u32,
    /// Total number of tokens consumed by this request.
    ///
    /// For embeddings, this is typically the same as `prompt_tokens`, unless the API
    /// changes how it reports usage data in the future.
    pub total_tokens: u32,
}

/// Creates embeddings using the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).
///
/// # Parameters
///
/// * `client` - The [`OpenAIClient`](crate::config::OpenAIClient) to use for the request.
/// * `request` - A [`CreateEmbeddingsRequest`] specifying the model and input(s).
///
/// # Returns
///
/// A [`CreateEmbeddingsResponse`] containing one or more embedding vectors.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: if the request fails at the network layer.
/// - [`OpenAIError::DeserializeError`]: if the response fails to parse.
/// - [`OpenAIError::APIError`]: if OpenAI returns an error (e.g., invalid request).
pub async fn create_embeddings(
    client: &OpenAIClient,
    request: &CreateEmbeddingsRequest,
) -> Result<CreateEmbeddingsResponse, OpenAIError> {
    // According to the OpenAI docs, the endpoint for embeddings is:
    // POST /v1/embeddings
    let endpoint = "embeddings";
    post_json(client, endpoint, request).await
}
