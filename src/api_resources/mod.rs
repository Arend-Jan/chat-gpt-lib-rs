//! # API Resources Module
//!
//! This module groups together the various API resource modules that correspond to
//! different OpenAI endpoints, such as models, completions, chat, embeddings, etc.
//! Each sub-module provides high-level functions and data structures for interacting
//! with a specific set of endpoints.
//!
//! ## Currently Implemented
//!
//! - [`models`]: Retrieve and list available models
//! - [`completions`]: Generate text completions
//! - [`chat`]: Handle chat-based completions (ChatGPT)
//! - [`embeddings`]: Obtain vector embeddings for text
//! - [`moderations`]: Check text for policy violations
//! - [`fine_tunes`]: Manage fine-tuning jobs
//! - [`files`]: Upload and manage files
//!
//! ## Planned Modules
//!
//!
//! # Example
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::OpenAIClient;
//! use chat_gpt_lib_rs::api_resources::models;
//! use chat_gpt_lib_rs::error::OpenAIError;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?;
//!
//!     // Example: list and retrieve models
//!     let model_list = models::list_models(&client).await?;
//!     let first_model = &model_list[0];
//!     let model_details = models::retrieve_model(&client, &first_model.id).await?;
//!     println!("Model details: {:?}", model_details);
//!
//!     Ok(())
//! }
//! ```

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod files;
pub mod fine_tunes;
/// Resources for working with OpenAI Models.
pub mod models;
pub mod moderations;
