//! This module provides functionality for creating text completions using the
//! [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions).
//!
//! **Note**: This struct (`CreateCompletionRequest`) has been expanded to capture additional
//! fields from the OpenAI specification, including `best_of`, `seed`, `suffix`, etc. Some
//! fields support multiple data types (e.g., `prompt`, `stop`) using `#[serde(untagged)]` enums
//! for flexible deserialization and serialization.
//!
//! # Overview
//!
//! The Completions API can generate or manipulate text based on a given prompt. You specify a model
//! (e.g., `"gpt-3.5-turbo-instruct"`), a prompt, and various parameters like `max_tokens` and
//! `temperature`.  
//! **Important**: This request object allows for advanced configurations such as
//! `best_of`, `seed`, and `logit_bias`. Use them carefully, especially if they can consume many
//! tokens or produce unexpected outputs.
//!
//! Typical usage involves calling [`create_completion`] with a [`CreateCompletionRequest`]:
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::completions::{create_completion, CreateCompletionRequest, PromptInput};
//! use chat_gpt_lib_rs::OpenAIClient;
//! use chat_gpt_lib_rs::error::OpenAIError;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY
//!
//!     let request = CreateCompletionRequest {
//!         model: "gpt-3.5-turbo-instruct".to_string(),
//!         // `PromptInput::String` variant if we just have a single prompt text
//!         prompt: Some(PromptInput::String("Tell me a joke about cats".to_string())),
//!         max_tokens: Some(50),
//!         temperature: Some(1.0),
//!         ..Default::default()
//!     };
//!
//!     let response = create_completion(&client, &request).await?;
//!     if let Some(choice) = response.choices.get(0) {
//!         println!("Completion: {}", choice.text);
//!     }
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::api::post_json;
use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// Represents the diverse ways a prompt can be supplied:
///
/// - A single string (`"Hello, world!"`)
/// - An array of strings
/// - An array of integers (token IDs)
/// - An array of arrays of integers (multiple sequences of token IDs)
///
/// This enumeration corresponds to the JSON schema's `oneOf` for `prompt`.  
/// By using `#[serde(untagged)]`, Serde will automatically handle whichever
/// variant is provided.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum PromptInput {
    /// A single string prompt
    String(String),
    /// Multiple string prompts
    Strings(Vec<String>),
    /// A single sequence of token IDs
    Ints(Vec<i64>),
    /// Multiple sequences of token IDs
    MultiInts(Vec<Vec<i64>>),
}

/// Represents the different ways `stop` can be supplied:
///
/// - A single string (e.g. `"\n"`)
/// - An array of up to 4 strings (e.g. `[".END", "Goodbye"]`)
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopSequence {
    /// A single stopping string
    Single(String),
    /// Multiple stopping strings
    Multiple(Vec<String>),
}

/// Placeholder for potential streaming options, per the spec reference:
/// `#/components/schemas/ChatCompletionStreamOptions`.
///
/// If you plan to implement streaming logic, define fields here accordingly.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ChatCompletionStreamOptions {
    // For now, this is an empty placeholder.
    // Extend or remove based on your streaming logic requirements.
}

/// A request struct for creating text completions with the OpenAI API.
///
/// This struct fully reflects the extended specification from OpenAI,
/// including fields such as `best_of`, `seed`, and `suffix`.
#[derive(Debug, Serialize, Default, Clone)]
pub struct CreateCompletionRequest {
    /// **Required.** ID of the model to use. For example: `"gpt-3.5-turbo-instruct"`, `"davinci-002"`,
    /// or `"text-davinci-003"`.
    pub model: String,

    /// **Required.** The prompt(s) to generate completions for.
    /// Defaults to `<|endoftext|>` if not provided.
    ///
    /// Can be a single string, an array of strings, an array of integers (token IDs),
    /// or an array of arrays of integers (multiple token sequences).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_prompt")]
    pub prompt: Option<PromptInput>,

    /// The maximum number of [tokens](https://platform.openai.com/tokenizer) to generate
    /// in the completion. Defaults to 16.
    ///
    /// The combined length of prompt + `max_tokens` cannot exceed the model's context length.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,

    /// What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the
    /// output more random, while lower values like `0.2` will make it more focused and deterministic.
    ///
    /// We generally recommend altering this or `top_p` but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_temperature")]
    pub temperature: Option<f64>,

    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with `top_p` probability mass. So `0.1` means only
    /// the tokens comprising the top 10% probability mass are considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_top_p")]
    pub top_p: Option<f64>,

    /// How many completions to generate for each prompt. Defaults to 1.
    ///
    /// **Note**: Because this parameter generates many completions, it can quickly consume your
    /// token quota. Use carefully and ensure you have reasonable settings for `max_tokens` and `stop`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_n")]
    pub n: Option<u32>,

    /// Generates `best_of` completions server-side and returns the "best" (the one with the
    /// highest log probability per token). Must be greater than `n`. Defaults to 1.
    ///
    /// **Note**: This parameter can quickly consume your token quota if `best_of` is large.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default = "default_best_of")]
    pub best_of: Option<u32>,

    /// Whether to stream back partial progress. Defaults to `false`.
    ///
    /// If set to `true`, tokens will be sent as data-only server-sent events (SSE) as they
    /// become available, with the stream terminated by a `data: [DONE]` message.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub stream: Option<bool>,

    /// Additional options that could be used in streaming scenarios.
    /// This is a placeholder for any extended streaming logic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,

    /// Include the log probabilities on the `logprobs` most likely tokens, along with the chosen tokens.
    /// A value of `5` returns the 5 most likely tokens. Defaults to `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Echo back the prompt in addition to the completion. Defaults to `false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub echo: Option<bool>,

    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence. Defaults to `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in
    /// the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub presence_penalty: Option<f64>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency
    /// in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub frequency_penalty: Option<f64>,

    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Maps token IDs to a bias value from -100 to 100. Defaults to `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i32>>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    /// This is optional, but recommended. Example: `"user-1234"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// If specified, the system will make a best effort to sample deterministically.
    /// Repeated requests with the same `seed` and parameters should return the same result (best-effort).
    ///
    /// Determinism is not guaranteed, and you should refer to the `system_fingerprint` in the response
    /// to monitor backend changes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// The suffix that comes after a completion of inserted text. This parameter is only supported
    /// for `gpt-3.5-turbo-instruct`. Defaults to `null`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

/// Default prompt is `<|endoftext|>`, per the specification.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_prompt() -> Option<PromptInput> {
    Some(PromptInput::String("<|endoftext|>".to_string()))
}

/// Default max_tokens is `16`.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_max_tokens() -> Option<u32> {
    Some(16)
}

/// Default temperature is `1.0`.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_temperature() -> Option<f64> {
    Some(1.0)
}

/// Default top_p is `1.0`.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_top_p() -> Option<f64> {
    Some(1.0)
}

/// Default `n` is `1`.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_n() -> Option<u32> {
    Some(1)
}

/// Default `best_of` is `1`.
#[allow(dead_code)] // This way, Serde can still invoke them at runtime, but the compiler won’t complain.
fn default_best_of() -> Option<u32> {
    Some(1)
}

/// The response returned by the OpenAI Completions API.
///
/// Contains generated `choices` plus optional `usage` metrics.
#[derive(Debug, Deserialize)]
pub struct CreateCompletionResponse {
    /// An identifier fo this completion (e.g. `"cmpl-xxxxxxxx"`).
    pub id: String,
    /// The object type, usually `"text_completion"`.
    pub object: String,
    /// The creation time in epoch seconds.
    pub created: u64,
    /// The model used for this request.
    pub model: String,
    /// A list of generated completions.
    pub choices: Vec<CompletionChoice>,
    /// Token usage data (optional field).
    #[serde(default)]
    pub usage: Option<CompletionUsage>,
}

/// A single generated completion choice within a [`CreateCompletionResponse`].
#[derive(Debug, Deserialize)]
pub struct CompletionChoice {
    /// The generated text.
    pub text: String,
    /// Which completion index this choice corresponds to (useful if `n` > 1).
    pub index: u32,
    /// The reason why the completion ended (e.g., "stop", "length").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// The log probabilities, if `logprobs` was requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Token usage data, if requested or included by default.
#[derive(Debug, Deserialize)]
pub struct CompletionUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the generated completion.
    pub completion_tokens: u32,
    /// Total number of tokens consumed by this request.
    pub total_tokens: u32,
}

/// Creates a text completion using the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions).
///
/// # Parameters
///
/// * `client` - The [`OpenAIClient`](crate::config::OpenAIClient) to use for the request.
/// * `request` - A [`CreateCompletionRequest`] specifying the prompt, model, and additional parameters.
///
/// # Returns
///
/// A [`CreateCompletionResponse`] containing the generated text (in [`CompletionChoice`])
/// and metadata about usage and indexing.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: if the request fails at the network layer.
/// - [`OpenAIError::DeserializeError`]: if the response fails to parse.
/// - [`OpenAIError::APIError`]: if OpenAI returns an error (e.g. invalid request).
pub async fn create_completion(
    client: &OpenAIClient,
    request: &CreateCompletionRequest,
) -> Result<CreateCompletionResponse, OpenAIError> {
    let endpoint = "completions";
    post_json(client, endpoint, request).await
}

#[cfg(test)]
mod tests {
    /// # Tests for the `completions` module
    ///
    /// These tests use [`wiremock`](https://crates.io/crates/wiremock) to mock responses from the
    /// `/v1/completions` endpoint. We cover:
    /// 1. A successful JSON response, ensuring we can deserialize a [`CreateCompletionResponse`].
    /// 2. A non-2xx OpenAI-style error, which should map to [`OpenAIError::APIError`].
    /// 3. Malformed JSON that triggers a [`OpenAIError::DeserializeError`].
    ///
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_create_completion_success() {
        // Spin up a local mock server
        let mock_server = MockServer::start().await;

        // Mock JSON body for a successful 200 response
        let success_body = json!({
            "id": "cmpl-12345",
            "object": "text_completion",
            "created": 1673643147,
            "model": "text-davinci-003",
            "choices": [{
                "text": "This is a funny cat joke!",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "total_tokens": 17
            }
        });

        // Expect a POST to /v1/completions
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            // Override the base URL to the mock server
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Create a minimal request
        let req = CreateCompletionRequest {
            model: "text-davinci-003".to_string(),
            prompt: Some(PromptInput::String("Tell me a cat joke".into())),
            max_tokens: Some(20),
            ..Default::default()
        };

        // Call the function under test
        let result = create_completion(&client, &req).await;
        assert!(result.is_ok(), "Expected success, got: {:?}", result);

        let resp = result.unwrap();
        assert_eq!(resp.id, "cmpl-12345");
        assert_eq!(resp.object, "text_completion");
        assert_eq!(resp.model, "text-davinci-003");
        assert_eq!(resp.choices.len(), 1);

        let choice = &resp.choices[0];
        assert_eq!(choice.text, "This is a funny cat joke!");
        assert_eq!(choice.index, 0);
        assert_eq!(choice.finish_reason.as_deref(), Some("stop"));

        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 17);
    }

    #[tokio::test]
    async fn test_create_completion_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Model is unavailable",
                "type": "invalid_request_error",
                "code": "model_unavailable"
            }
        });

        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateCompletionRequest {
            model: "unknown-model".into(),
            prompt: Some(PromptInput::String("Hello".into())),
            ..Default::default()
        };

        let result = create_completion(&client, &req).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Model is unavailable"));
            }
            other => panic!("Expected APIError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_create_completion_deserialize_error() {
        let mock_server = MockServer::start().await;

        // Return a 200 but with malformed JSON that doesn't match `CreateCompletionResponse`
        let malformed_json = r#"{
            "id": "cmpl-12345",
            "object": "text_completion",
            "created": "invalid_number",
            "model": "text-davinci-003",
            "choices": "should_be_array"
        }"#;

        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(malformed_json, "application/json"),
            )
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateCompletionRequest {
            model: "text-davinci-003".into(),
            prompt: Some(PromptInput::String("Hello".into())),
            ..Default::default()
        };

        let result = create_completion(&client, &req).await;
        match result {
            Err(OpenAIError::DeserializeError(_)) => {
                // success
            }
            other => panic!("Expected DeserializeError, got {:?}", other),
        }
    }
}
