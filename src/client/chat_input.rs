use crate::client::tool::{Tool, ToolChoice};
use crate::models::{LogitBias, Model, Role};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the input for the chat API call.
///
/// This struct is used to configure various parameters for generating a chat completion using the API.
///
/// Fields:
/// - `model`: Specifies the ID of the model to use for generating the chat completion.
/// - `messages`: A list of messages comprising the conversation so far.
/// - `temperature`: (Optional) Sampling temperature to use, between 0 and 2. Higher values make the output more random.
/// - `top_p`: (Optional) Nucleus sampling parameter. Consider the tokens with the top p probability mass.
/// - `n`: (Optional) Number of chat completion choices to generate for each input message.
/// - `stream`: (Optional) If set, partial message deltas will be sent as they become available.
/// - `stop`: (Optional) Up to 4 sequences where the API will stop generating further tokens.
/// - `max_tokens`: (Optional) The maximum number of tokens that can be generated in the chat completion.
/// - `presence_penalty`: (Optional) Penalize new tokens based on whether they appear in the text so far.
/// - `frequency_penalty`: (Optional) Penalize new tokens based on their existing frequency in the text so far.
/// - `logit_bias`: (Optional) Modify the likelihood of specified tokens appearing in the completion.
/// - `user`: (Optional) A unique identifier representing your end-user for monitoring and abuse detection.
/// - `logprobs`: (Optional) Whether to return log probabilities of the output tokens.
/// - `top_logprobs`: (Optional) Number of most likely tokens to return at each token position with log probabilities.
/// - `response_format`: (Optional) Specifies the format that the model must output (e.g., JSON).
/// - `seed`: (Optional) A seed for deterministic sampling. Repeated requests with the same seed should return the same result.
/// - `stream_options`: (Optional) Options for streaming response. Only set this when stream is true.
/// - `tools`: (Optional) A list of tools the model may call. Currently, only functions are supported as tools.
/// - `tool_choice`: (Optional) Controls which tool is called by the model. Can be none, auto, or a specific tool.
#[derive(Debug, Serialize)]
pub struct ChatInput {
    pub model: Model,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<LogitBias>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

impl Default for ChatInput {
    fn default() -> Self {
        Self {
            model: Model::Gpt_4,
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            seed: None,
            stream_options: None,
            tools: None,
            tool_choice: None,
        }
    }
}

/// Represents a message in the chat API call.
///
/// Fields:
/// - `role`: The role of the message sender (e.g., system, user, assistant).
/// - `content`: The content of the message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    #[serde(flatten)]
    pub content: MessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum MessageContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

impl From<String> for MessageContent {
    /// Converts a `String` to `MessageContent::Text`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to convert.
    ///
    fn from(text: String) -> Self {
        MessageContent::Text { text }
    }
}

impl From<&str> for MessageContent {
    /// Converts a `&str` to `MessageContent::Text`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to convert.
    ///
    fn from(text: &str) -> Self {
        MessageContent::Text {
            text: text.to_string(),
        }
    }
}

impl From<(Role, String)> for Message {
    /// Converts a tuple of `(Role, String)` to a `Message`.
    ///
    /// # Arguments
    ///
    /// * `(role, text)` - A tuple where `role` is the role of the message sender and `text` is the text content.
    ///
    fn from((role, text): (Role, String)) -> Self {
        Self {
            role,
            content: MessageContent::Text { text },
        }
    }
}

impl From<(Role, &str)> for Message {
    /// Converts a tuple of `(Role, &str)` to a `Message`.
    ///
    /// # Arguments
    ///
    /// * `(role, text)` - A tuple where `role` is the role of the message sender and `text` is the text content.
    ///
    fn from((role, text): (Role, &str)) -> Self {
        Self {
            role,
            content: MessageContent::Text {
                text: text.to_string(),
            },
        }
    }
}

impl From<String> for Message {
    /// Converts a `String` to a `Message` with `Role::User`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to convert.
    ///
    fn from(text: String) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::from(text),
        }
    }
}

impl From<&str> for Message {
    /// Converts a `&str` to a `Message` with `Role::User`.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to convert.
    ///
    fn from(text: &str) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::from(text),
        }
    }
}

impl fmt::Display for MessageContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageContent::Text { text } => write!(f, "Text: {}", text),
            MessageContent::ImageUrl { image_url } => write!(f, "Image URL: {}", image_url.url),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Role;

    #[test]
    fn test_chat_input_default() {
        let chat_input = ChatInput::default();
        assert_eq!(chat_input.model, Model::Gpt_4);
        assert!(chat_input.messages.is_empty());
        assert!(chat_input.temperature.is_none());
        assert!(chat_input.top_p.is_none());
        assert!(chat_input.n.is_none());
        assert!(chat_input.stream.is_none());
        assert!(chat_input.stop.is_none());
        assert!(chat_input.max_tokens.is_none());
        assert!(chat_input.presence_penalty.is_none());
        assert!(chat_input.frequency_penalty.is_none());
        assert!(chat_input.logit_bias.is_none());
        assert!(chat_input.user.is_none());
        assert!(chat_input.logprobs.is_none());
        assert!(chat_input.top_logprobs.is_none());
        assert!(chat_input.response_format.is_none());
        assert!(chat_input.seed.is_none());
        assert!(chat_input.stream_options.is_none());
        assert!(chat_input.tools.is_none());
        assert!(chat_input.tool_choice.is_none());
    }

    #[test]
    fn test_message_struct() {
        let message = Message {
            role: Role::User,
            content: MessageContent::Text {
                text: "Hello, how can I help you?".into(),
            },
        };
        assert_eq!(message.role, Role::User);
        assert_eq!(
            message.content,
            MessageContent::Text {
                text: "Hello, how can I help you?".to_string(),
            }
        );
    }
}
