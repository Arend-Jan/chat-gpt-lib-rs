//! This module provides functionality for classifying text against OpenAI's content moderation
//! policies using the [OpenAI Moderations API](https://platform.openai.com/docs/api-reference/moderations).
//!
//! The Moderations API takes text (or multiple pieces of text) and returns a set of boolean flags
//! indicating whether the content violates certain categories (e.g., hate, self-harm, sexual), along
//! with confidence scores for each category.
//!
//! # Overview
//!
//! You can call [`create_moderation`] with a [`CreateModerationRequest`], specifying your input text(s)
//! (and optionally, a specific model). The response (`CreateModerationResponse`) includes a list of
//! [`ModerationResult`] objects—one per input. Each result contains a set of [`ModerationCategories`],
//! matching confidence scores ([`ModerationCategoryScores`]), and a `flagged` boolean indicating if the
//! text violates policy overall.
//!
//! # Example
//!
//! ```rust
//! use chat_gpt_lib_rs::api_resources::moderations::{create_moderation, CreateModerationRequest, ModerationsInput};
//! use chat_gpt_lib_rs::error::OpenAIError;
//! use chat_gpt_lib_rs::OpenAIClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     // load environment variables from a .env file, if present (optional).
//!     dotenvy::dotenv().ok();
//!
//!     let client = OpenAIClient::new(None)?;
//!     let request = CreateModerationRequest {
//!         input: ModerationsInput::String("I hate you and want to harm you.".to_string()),
//!         model: None, // or Some("text-moderation-latest".to_string())
//!     };
//!
//!     let response = create_moderation(&client, &request).await?;
//!     for (i, result) in response.results.iter().enumerate() {
//!         println!("== Result {} ==", i);
//!         println!("Flagged: {}", result.flagged);
//!         println!("Hate category: {}", result.categories.hate);
//!         println!("Hate score: {}", result.category_scores.hate);
//!         // ...and so on for other categories
//!     }
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::api::post_json;
use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// Represents the multiple ways the input can be supplied for moderations:
///
/// - A single string
/// - An array of strings
///
/// Other forms (such as token arrays) are not commonly used for this endpoint.
/// If you need a more advanced setup, you can adapt this or add variants as needed.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ModerationsInput {
    /// A single string input
    String(String),
    /// Multiple string inputs
    Strings(Vec<String>),
}

/// A request struct for creating a moderation check using the OpenAI Moderations API.
///
/// For more details, see the [API documentation](https://platform.openai.com/docs/api-reference/moderations).
#[derive(Debug, Serialize, Clone)]
pub struct CreateModerationRequest {
    /// The input text(s) to classify.  
    /// **Required** by the API.
    pub input: ModerationsInput,

    /// *Optional.* Two possible values are often used:  
    /// - `"text-moderation-stable"`  
    /// - `"text-moderation-latest"`  
    ///
    /// If omitted, the default model is used.  
    /// See [OpenAI's docs](https://platform.openai.com/docs/api-reference/moderations) for details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// The response returned by the OpenAI Moderations API.
///
/// Contains an identifier and a list of [`ModerationResult`] items corresponding to each input.
#[derive(Debug, Deserialize)]
pub struct CreateModerationResponse {
    /// An identifier for this moderation request (e.g., "modr-xxxxxx").
    pub id: String,
    /// The moderation model used.
    pub model: String,
    /// A list of moderation results—one per input in `CreateModerationRequest.input`.
    pub results: Vec<ModerationResult>,
}

/// A single moderation result, indicating how the input text matches various policy categories.
#[derive(Debug, Deserialize)]
pub struct ModerationResult {
    /// Boolean flags indicating which categories (hate, self-harm, sexual, etc.) are triggered.
    pub categories: ModerationCategories,
    /// Floating-point confidence scores for each category.
    pub category_scores: ModerationCategoryScores,
    /// Overall flag indicating if the content violates policy (i.e., if the text should be disallowed).
    pub flagged: bool,
}

/// A breakdown of the moderation categories.
///
/// Each field corresponds to a distinct policy category recognized by OpenAI's model.
/// If `true`, the text has been flagged under that category.
#[derive(Debug, Deserialize)]
pub struct ModerationCategories {
    /// Hateful content directed towards a protected group or individual.
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    /// Hateful content with threats.
    pub hate_threatening: bool,
    #[serde(rename = "self-harm")]
    /// Content about self-harm or suicide.
    pub self_harm: bool,
    /// If `true`, the text includes sexual content or references.
    pub sexual: bool,
    #[serde(rename = "sexual/minors")]
    /// If `true`, the text includes sexual content involving minors.
    pub sexual_minors: bool,
    /// If `true`, the text includes violent content or context.
    pub violence: bool,
    #[serde(rename = "violence/graphic")]
    /// If `true`, the text includes particularly graphic or gory violence.
    pub violence_graphic: bool,
}

/// Floating-point confidence scores for each moderated category.
///
/// Higher values indicate higher model confidence that the content falls under that category.
#[derive(Debug, Deserialize)]
pub struct ModerationCategoryScores {
    /// The confidence score for hateful content.
    pub hate: f64,
    #[serde(rename = "hate/threatening")]
    /// The confidence score for hateful content that includes threats.
    pub hate_threatening: f64,
    #[serde(rename = "self-harm")]
    /// The confidence score for self-harm or suicidal content.
    pub self_harm: f64,
    /// The confidence score for sexual content or references.
    pub sexual: f64,
    #[serde(rename = "sexual/minors")]
    /// The confidence score for sexual content involving minors.
    pub sexual_minors: f64,
    /// The confidence score for violent content or context.
    pub violence: f64,
    #[serde(rename = "violence/graphic")]
    /// The confidence score for particularly graphic or gory violence.
    pub violence_graphic: f64,
}

/// Creates a moderation request using the [OpenAI Moderations API](https://platform.openai.com/docs/api-reference/moderations).
///
/// # Parameters
///
/// * `client` - The [`OpenAIClient`](crate::config::OpenAIClient) to use for the request.
/// * `request` - A [`CreateModerationRequest`] specifying the input text(s) and an optional model.
///
/// # Returns
///
/// A [`CreateModerationResponse`] containing moderation results for each input.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: if the request fails at the network layer.
/// - [`OpenAIError::DeserializeError`]: if the response fails to parse.
/// - [`OpenAIError::APIError`]: if OpenAI returns an error (e.g. invalid request).
pub async fn create_moderation(
    client: &OpenAIClient,
    request: &CreateModerationRequest,
) -> Result<CreateModerationResponse, OpenAIError> {
    // POST /v1/moderations
    let endpoint = "moderations";
    post_json(client, endpoint, request).await
}
