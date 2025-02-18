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
//!         model: None, // or Some("text-moderation-latest".into())
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

use super::models::Model;

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
    pub model: Option<Model>,
}

/// The response returned by the OpenAI Moderations API.
///
/// Contains an identifier and a list of [`ModerationResult`] items corresponding to each input.
#[derive(Debug, Deserialize)]
pub struct CreateModerationResponse {
    /// An identifier for this moderation request (e.g., "modr-xxxxxx").
    pub id: String,
    /// The moderation model used.
    pub model: Model,
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

#[cfg(test)]
mod tests {
    /// # Tests for the `moderations` module
    ///
    /// These tests use [`wiremock`](https://crates.io/crates/wiremock) to simulate the
    /// [OpenAI Moderations API](https://platform.openai.com/docs/api-reference/moderations).
    /// We specifically test [`create_moderation`] for:
    /// 1. A **success** scenario where it returns a valid [`CreateModerationResponse`].
    /// 2. An **API error** scenario with a non-2xx response.
    /// 3. A **deserialization** scenario where JSON is malformed.
    ///
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_create_moderation_success() {
        // Start a local mock server
        let mock_server = MockServer::start().await;

        // Example success response
        let success_body = json!({
            "id": "modr-abc123",
            "model": "text-moderation-latest",
            "results": [
                {
                    "flagged": true,
                    "categories": {
                        "hate": true,
                        "hate/threatening": false,
                        "self-harm": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": true,
                        "violence/graphic": false
                    },
                    "category_scores": {
                        "hate": 0.98,
                        "hate/threatening": 0.25,
                        "self-harm": 0.05,
                        "sexual": 0.0,
                        "sexual/minors": 0.0,
                        "violence": 0.85,
                        "violence/graphic": 0.1
                    }
                }
            ]
        });

        // Mock a 200 response on /v1/moderations
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        // Build a test client
        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Minimal request
        let req = CreateModerationRequest {
            input: ModerationsInput::String("some potentially hateful text".to_string()),
            model: Some("text-moderation-latest".into()),
        };

        let result = create_moderation(&client, &req).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let resp = result.unwrap();
        assert_eq!(resp.id, "modr-abc123");
        assert_eq!(resp.model, "text-moderation-latest".into());
        assert_eq!(resp.results.len(), 1);

        let first = &resp.results[0];
        assert!(first.flagged);
        assert!(first.categories.hate);
        assert!(first.categories.violence);
        assert!(!first.categories.hate_threatening);
        assert_eq!(first.category_scores.hate, 0.98);
        assert_eq!(first.category_scores.violence, 0.85);
    }

    #[tokio::test]
    async fn test_create_moderation_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Invalid model",
                "type": "invalid_request_error",
                "code": null
            }
        });

        // Mock a 400 error on POST /moderations
        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(ResponseTemplate::new(400).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateModerationRequest {
            input: ModerationsInput::Strings(vec!["test text".into()]),
            model: Some("text-moderation-unknown".into()),
        };

        let result = create_moderation(&client, &req).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Invalid model"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_create_moderation_deserialize_error() {
        let mock_server = MockServer::start().await;

        // Return 200 with malformed JSON
        let malformed_body = r#"{
          "id": "modr-12345",
          "model": "text-moderation-latest",
          "results": "should_be_array_not_string"
        }"#;

        Mock::given(method("POST"))
            .and(path("/moderations"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(malformed_body, "application/json"),
            )
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateModerationRequest {
            input: ModerationsInput::String("Another text".to_string()),
            model: None,
        };

        let result = create_moderation(&client, &req).await;
        match result {
            Err(OpenAIError::DeserializeError(_)) => {
                // success
            }
            other => panic!("Expected DeserializeError, got {:?}", other),
        }
    }
}
