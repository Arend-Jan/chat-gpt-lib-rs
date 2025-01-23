//! This module provides functionality for creating chat-based completions using the
//! [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
//!
//! The Chat API is designed for conversational interactions, where each request includes a list
//! of messages with a role (system, user, or assistant). The model responds based on the context
//! established by these messages, allowing for more interactive and context-aware responses
//! compared to plain completions.
//!
//! # Overview
//!
//! The core usage involves calling [`create_chat_completion`] with a [`CreateChatCompletionRequest`],
//! which includes a sequence of [`ChatMessage`] items. Each `ChatMessage` has a `role` and `content`.
//! The API then returns a [`CreateChatCompletionResponse`] containing one or more
//! [`ChatCompletionChoice`] objects (depending on the `n` parameter).
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::chat::{create_chat_completion, CreateChatCompletionRequest, ChatMessage, ChatRole};
//! use chat_gpt_lib_rs::error::OpenAIError;
//! use chat_gpt_lib_rs::OpenAIClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY
//!
//!     let request = CreateChatCompletionRequest {
//!         model: "o1-mini".to_string(),
//!         messages: vec![
//!             ChatMessage {
//!                 role: ChatRole::System,
//!                 content: "You are a helpful assistant.".to_string(),
//!                 name: None,
//!             },
//!             ChatMessage {
//!                 role: ChatRole::User,
//!                 content: "Write a tagline for an ice cream shop.".to_string(),
//!                 name: None,
//!             },
//!         ],
//!         max_tokens: Some(50),
//!         temperature: Some(0.7),
//!         ..Default::default()
//!     };
//!
//!     let response = create_chat_completion(&client, &request).await?;
//!
//!     for choice in &response.choices {
//!         println!("Assistant: {}", choice.message.content);
//!     }
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::post_json;
use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// The role of a message in the chat sequence.
///
/// Typically one of `system`, `user`, `assistant`. OpenAI may add or adjust roles in the future.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// For system-level instructions (e.g. "You are a helpful assistant.")
    System,
    /// For user-supplied messages
    User,
    /// For assistant messages (responses from the model)
    Assistant,
    /// For tools
    Tool,
    /// For function
    Function,
    /// Experimental or extended role types, if they become available
    #[serde(other)]
    Other,
}

/// A single message in a chat conversation.
///
/// Each message has:
/// - A [`ChatRole`], indicating who is sending the message (system, user, assistant).
/// - The message `content`.
/// - An optional `name` for the user or system, if applicable.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    /// The role of the sender (system, user, or assistant).
    pub role: ChatRole,
    /// The content of the message.
    pub content: String,
    /// The (optional) name of the user or system. This can be used to identify
    /// the speaker when multiple users or participants exist in a conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A request struct for creating chat completions with the OpenAI Chat Completions API.
///
/// # Fields
/// - `model`: The ID of the model to use (e.g., "gpt-3.5-turbo").
/// - `messages`: A list of [`ChatMessage`] items providing the conversation history.
/// - `stream`: Whether or not to stream responses via server-sent events.
/// - `max_tokens`, `temperature`, `top_p`, etc.: Parameters controlling the generation.
/// - `n`: Number of chat completion choices to generate.
/// - `logit_bias`, `user`: Additional advanced parameters.
#[derive(Debug, Serialize, Default, Clone)]
pub struct CreateChatCompletionRequest {
    /// **Required**. The model used for this chat request.
    /// Examples: "gpt-3.5-turbo", "gpt-4".
    pub model: String,

    /// **Required**. The messages that make up the conversation so far.
    pub messages: Vec<ChatMessage>,

    /// Controls the creativity of the output. 0 is the most deterministic, 2 is the most creative.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// The nucleus sampling parameter. Like `temperature`, but a value like 0.1 means only
    /// the top 10% probability mass is considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// How many chat completion choices to generate for each input message. Defaults to 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// If set, partial message deltas are sent as data-only server-sent events (SSE) as they become available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// The maximum number of tokens allowed for the generated answer. Defaults to the max tokens allowed by the model minus the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// A map between token (encoded as a string) and an associated bias from -100 to 100
    /// that adjusts the likelihood of the token appearing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i32>>,

    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// The response returned by the OpenAI Chat Completions API.
///
/// Includes one or more chat-based completion choices and any usage statistics.
#[derive(Debug, Deserialize)]
pub struct CreateChatCompletionResponse {
    /// An identifier for this chat completion (e.g., "chatcmpl-xxxxxx").
    pub id: String,
    /// The object type, usually "chat.completion".
    pub object: String,
    /// The creation time in epoch seconds.
    pub created: u64,
    /// The base model used for this request.
    pub model: String,
    /// A list of generated chat completion choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Token usage data (optional field).
    #[serde(default)]
    pub usage: Option<ChatCompletionUsage>,
}

/// A single chat completion choice within a [`CreateChatCompletionResponse`].
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChoice {
    /// The index of this choice (useful if `n` > 1).
    pub index: u32,
    /// The chat message object containing the role and content.
    pub message: ChatMessage,
    /// Why the chat completion ended (e.g., "stop", "length").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Token usage data, if requested or included by default.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionUsage {
    /// Number of tokens used in the prompt so far.
    pub prompt_tokens: u32,
    /// Number of tokens used in the generated answer.
    pub completion_tokens: u32,
    /// Total number of tokens consumed by this request.
    pub total_tokens: u32,
}

/// Creates a chat-based completion using the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).
///
/// # Parameters
/// * `client` - The [`OpenAIClient`](crate::config::OpenAIClient) to use for the request.
/// * `request` - A [`CreateChatCompletionRequest`] specifying the messages, model, and other parameters.
///
/// # Returns
/// A [`CreateChatCompletionResponse`] containing one or more [`ChatCompletionChoice`] items.
///
/// # Errors
/// - [`OpenAIError::HTTPError`]: if the request fails at the network layer.
/// - [`OpenAIError::DeserializeError`]: if the response fails to parse.
/// - [`OpenAIError::APIError`]: if OpenAI returns an error (e.g., invalid request).
pub async fn create_chat_completion(
    client: &OpenAIClient,
    request: &CreateChatCompletionRequest,
) -> Result<CreateChatCompletionResponse, OpenAIError> {
    // According to the OpenAI docs, the endpoint for chat completions is:
    // POST /v1/chat/completions
    let endpoint = "chat/completions";
    post_json(client, endpoint, request).await
}
