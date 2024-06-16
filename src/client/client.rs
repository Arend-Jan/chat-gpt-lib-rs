use log::debug;
use reqwest::{Client, StatusCode};

use super::{chat_input::ChatInput, chat_response::ChatResponse, error::ChatGPTError};

/// Main ChatGPTClient struct.
pub struct ChatGPTClient {
    base_url: String,
    api_key: String,
    client: Client,
}

impl ChatGPTClient {
    /// Creates a new ChatGPTClient with the given API key and base URL.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for the ChatGPT API.
    /// * `base_url` - The base URL for the ChatGPT API.
    pub fn new(api_key: &str, base_url: &str) -> Self {
        let client = Client::builder()
            .use_rustls_tls()
            .build()
            .expect("New client");

        Self {
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            client,
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
    /// use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model, Role};
    ///
    /// async fn example() {
    ///     let chat_gpt = ChatGPTClient::new("your_api_key", "https://api.openai.com");
    ///     let input = ChatInput {
    ///         model: Model::Gpt_4,
    ///         messages: vec![
    ///             Message {
    ///                 role: Role::System,
    ///                 content: "You are a helpful assistant.".into(),
    ///             },
    ///             Message {
    ///                 role: Role::User,
    ///                 content: "Who is the best field hockey player in the world".into(),
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
                "Request failed with status code: {status_code}\nHeaders: {headers:?}\nBody: {body}"
            );
            Err(ChatGPTError::RequestFailed(error_message))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::chat_response;
    use crate::Message;
    use crate::Model;
    use crate::Role;
    use reqwest::StatusCode;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_new_chat_gpt_client() {
        let api_key = "test_api_key";
        let base_url = "https://api.openai.com";

        let client = ChatGPTClient::new(api_key, base_url);

        assert_eq!(client.base_url, base_url);
        assert_eq!(client.api_key, api_key);
    }

    #[tokio::test]
    async fn test_chat_success() {
        let mock_server = MockServer::start().await;

        let client = ChatGPTClient::new("test_api_key", &mock_server.uri());

        let chat_input = ChatInput {
            model: Model::Gpt_4,
            messages: vec![
                Message {
                    role: Role::System,
                    content: "You are a helpful assistant.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Who is the best field hockey player in the world".into(),
                },
            ],
            ..Default::default()
        };

        let chat_response = ChatResponse {
            id: "test_id".to_string(),
            object: "chat.completion".to_string(),
            created: 123456789,
            model: Model::Gpt_4.to_string(),
            choices: vec![],
            usage: chat_response::Usage {
                completion_tokens: 42,
                prompt_tokens: 42,
                total_tokens: 84,
            },
        };

        Mock::given(path("/v1/chat/completions"))
            .and(method("POST"))
            .and(header("Authorization", "Bearer test_api_key"))
            .respond_with(ResponseTemplate::new(StatusCode::OK).set_body_json(&chat_response))
            .mount(&mock_server)
            .await;

        let response = client.chat(chat_input).await.unwrap();
        assert_eq!(response.id, chat_response.id);
        assert_eq!(response.object, chat_response.object);
    }

    #[tokio::test]
    async fn test_chat_failure() {
        let mock_server = MockServer::start().await;

        let client = ChatGPTClient::new("test_api_key", &mock_server.uri());

        let chat_input = ChatInput {
            model: Model::Gpt_4,
            messages: vec![
                Message {
                    role: Role::System,
                    content: "You are a helpful assistant.".into(),
                },
                Message {
                    role: Role::User,
                    content: "Who is the best field hockey player in the world".into(),
                },
            ],
            ..Default::default()
        };

        Mock::given(path("/v1/chat/completions"))
            .and(method("POST"))
            .and(header("Authorization", "Bearer test_api_key"))
            .respond_with(ResponseTemplate::new(StatusCode::BAD_REQUEST))
            .mount(&mock_server)
            .await;

        let result = client.chat(chat_input).await;
        assert!(result.is_err());
    }
}
