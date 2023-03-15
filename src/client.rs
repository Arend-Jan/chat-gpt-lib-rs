use crate::models::LogitBias;
use crate::models::Model;
use log::debug;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Main ChatGPTClient struct.
pub struct ChatGPTClient {
    base_url: String,
    api_key: String,
    client: Client,
}

/// Represents the input for the chat API call.
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
}

impl Default for ChatInput {
    fn default() -> Self {
        Self {
            model: Model::Gpt_4, // Set the default model
            messages: Vec::new(), // Set an empty vector for messages
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
        }
    }
}

/// Represents the response from the chat API call.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub usage: Usage,
    pub choices: Vec<Choice>,
}

/// Represents the usage information in the chat API response.
#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

/// Represents a choice in the chat API response.
#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: String,
}

/// Represents a message in the chat API call.
#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Enum representing possible errors in the ChatGPTClient.
#[derive(Debug)]
pub enum ChatGPTError {
    RequestFailed(String),
    Reqwest(reqwest::Error),
}

impl fmt::Display for ChatGPTError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChatGPTError::RequestFailed(message) => write!(f, "{}", message),
            ChatGPTError::Reqwest(error) => write!(f, "Reqwest error: {}", error),
        }
    }
}

impl std::error::Error for ChatGPTError {}

impl From<reqwest::Error> for ChatGPTError {
    fn from(error: reqwest::Error) -> Self {
        ChatGPTError::Reqwest(error)
    }
}

impl ChatGPTClient {
    /// Creates a new ChatGPTClient with the given API key and base URL.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for the ChatGPT API.
    /// * `base_url` - The base URL for the ChatGPT API.
    pub fn new(api_key: &str, base_url: &str) -> Self {
        env_logger::init();
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            client: Client::new(),
        }
    }

    /// Sends a request to the ChatGPT API with the given input and returns the response.
    ///
    /// # Arguments
    ///
    /// * `input` - A ChatInput struct representing the input for the chat API call.
    ///
    /// # Examples
    ///
    /// ```
    /// use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model};
    ///
    /// async fn example() {
    ///     let chat_gpt = ChatGPTClient::new("your_api_key", "https://api.openai.com");
    ///     let input = ChatInput {
    ///         model: Model::Gpt_4,
    ///         messages: vec![
    ///             Message {
    ///                 role: "system".to_string(),
    ///                 content: "You are a helpful assistant.".to_string(),
    ///             },
    ///             Message {
    ///                 role: "user".to_string(),
    ///                 content: "Who won the world series in 2020?".to_string(),
    ///             },
    ///         ],
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = chat_gpt.chat(input).await.unwrap();
    /// }
    /// ```
    /// # Errors
    ///
    /// Returns a ChatGPTError if the request fails.
    pub async fn chat(&self, input: ChatInput) -> Result<ChatResponse, ChatGPTError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&input)
            .send()
            .await?;

        debug!(
            "API call to url: {}\n with json payload: {:?}",
            &url, &input
        );

        // Check if the status code is 200
        if response.status() == StatusCode::OK {
            response
                .json::<ChatResponse>()
                .await
                .map_err(ChatGPTError::from)
        } else {
            let status_code = response.status();
            let headers = response.headers().clone();
            let body = response.text().await?;

            let error_message = format!(
                "Request failed with status code: {}\nHeaders: {:?}\nBody: {}",
                status_code, headers, body
            );
            Err(ChatGPTError::RequestFailed(error_message))
        }
    }
}
