use crate::models::Model;
use crate::models::LogitBias;
use reqwest::{Client, Response, Error};
use serde::{Deserialize, Serialize};

pub struct ChatGPTClient {
    base_url: String,
    api_key: String,
    client: Client,
}

#[derive(Debug, Serialize)]
pub struct ChatInput {
    pub model: Model,
    pub messages: Message,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub n: Option<usize>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub max_tokens: Option<usize>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub logit_bias: Option<LogitBias>,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub usage: Usage,
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl ChatGPTClient {
    pub fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            client: Client::new(),
        }
    }

    pub async fn chat(&self, input: ChatInput) -> Result<ChatResponse, Error> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&input)
            .send()
            .await?;

        response.json::<ChatResponse>().await
    }
}

