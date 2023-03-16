/// The `client` module provides the main functionality for interacting with the ChatGPT API.
pub mod client;

/// The `models` module contains the core data structures and types used in the ChatGPT API.
pub mod models;

// Re-export the main structs and enums for easier usage
pub use client::{ChatGPTClient, ChatInput, ChatResponse, Message};
pub use models::{LogitBias, Model, Role};

/// The `chat_gpt_lib_rs` crate provides a Rust interface to interact with the ChatGPT API.
///
/// The main functionality is provided by the `ChatGPTClient` struct, which allows you to send requests to the ChatGPT API.
/// The `models` module contains the core data structures and types used in the ChatGPT API.
///
/// # Examples
///
/// ```
/// use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model, Role};
///
/// async fn example() {
///     let chat_gpt = ChatGPTClient::new("your_api_key", "https://api.openai.com");
///     let input = ChatInput {
///         model: Model::Gpt3_5Turbo,
///         messages: vec![
///             Message {
///                 role: Role::System,
///                 content: "You are a helpful assistant.".to_string(),
///             },
///             Message {
///                 role: Role::User,
///                 content: "Who is the best field hockey player in the world".to_string(),
///             },
///         ],
///         ..Default::default()
///     };
///
///     let response = chat_gpt.chat(input).await.unwrap();
/// }
/// ```

