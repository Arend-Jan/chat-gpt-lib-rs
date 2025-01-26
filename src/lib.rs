#![warn(missing_docs)]
//! # OpenAI Rust Client Library
//!
//! This crate provides a Rust client library for the [OpenAI API](https://platform.openai.com/docs/api-reference),
//! implemented in an idiomatic Rust style. It aims to mirror the functionality of other official and
//! community OpenAI client libraries, while leveraging Rust’s strong type system, async
//! capabilities, and error handling.
//!
//! ## Getting Started
//!
//! Add the following to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! chat-gpt-lib-rs = "" // latest and greatest version
//! ```
//!
//! Then in your code:
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::{OpenAIClient, OpenAIError};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     // Create the client (pull the API key from the OPENAI_API_KEY environment variable by default).
//!     let client = OpenAIClient::new(Some("sk-...".to_string()))?;
//!
//!     // Now you can make calls like:
//!     // let response = client.create_completion("text-davinci-003", "Hello, world!", 50, 0.7).await?;
//!     // println!("Completion: {}", response);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Environment Variables
//!
//! By default, this library reads your OpenAI API key from the `OPENAI_API_KEY` environment variable.
//! If you wish to pass in a key at runtime, use the [`OpenAIClient::new`](crate::OpenAIClient::new) constructor
//! with an explicit key.
//!
//! ## Features
//!
//! - **Async-first** – Uses [Tokio](https://tokio.rs/) and [Reqwest](https://crates.io/crates/reqwest)
//! - **JSON Serialization** – Powered by [Serde](https://serde.rs/)
//! - **Custom Error Handling** – Utilizes [thiserror](https://crates.io/crates/thiserror) for ergonomic error types
//! - **Configurable** – Customize timeouts, organization IDs, or other settings
//!
//! ## Roadmap
//!
//! 1. Implement all major endpoints (e.g. Completions, Chat, Embeddings, Files, Fine-tunes, Moderations).
//! 2. Provide detailed logging with [`log`](https://crates.io/crates/log) and [`env_logger`](https://crates.io/crates/env_logger).
//! 3. Offer improved error handling by parsing OpenAI error fields directly.
//! 4. Provide thorough documentation and examples.
//!
//! ## Contributing
//!
//! Contributions to this project are more than welcome! Feel free to open issues, submit pull requests,
//! or suggest improvements. Please see our [GitHub repository](https://github.com/Arend-Jan/chat-gpt-lib-rs) for more details.

/// This module will contain the central configuration, including the `OpenAIClient` struct,
/// environment variable helpers, and other utilities.
pub mod config;

/// This module will define custom errors and error types for the library.
pub mod error;

/// This module will contain the low-level request and response logic,
/// leveraging `reqwest` to communicate with the OpenAI endpoints.
pub mod api;
pub mod api_resources;

/// Re-export commonly used structs and errors for convenience.
pub use config::OpenAIClient;
pub use error::OpenAIError;
