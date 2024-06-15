//! # chat_gpt_lib_rs
//!
//! The `chat_gpt_lib_rs` crate provides a Rust interface to interact with the ChatGPT API.
//!
//! This crate exports the following main items:
//!
//! - [`ChatGPTClient`]: Represents the main client to interact with the ChatGPT API.
//! - [`ChatInput`]: Represents the input for the chat API call.
//! - [`ChatResponse`]: Represents the response from the chat API call.
//! - [`Message`]: Represents a message in the chat API call.
//! - [`Model`]: Represents the available OpenAI models.
//! - [`Role`]: Represents the role of a message in the chat API call.
//! - [`LogitBias`]: Represents the logit bias used in API calls.
//! - [`count_tokens`]: Provides a rough estimation of the number of tokens in a given text.
//! For examples and more detailed usage information, please refer to the documentation of each exported item.

pub mod client;
pub mod models;
pub mod tokenizer;

pub use client::chat_input::{ChatInput, Message};
pub use client::chat_response::ChatResponse;
pub use client::client::ChatGPTClient;
pub use models::{LogitBias, Model, Role};
pub use tokenizer::count_tokens;
