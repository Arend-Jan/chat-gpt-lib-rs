//! The `error` module defines all error types that may arise when interacting with the OpenAI API.
//!
//! The main error type exported here is [`OpenAIError`], which enumerates various errors such as
//! configuration issues, HTTP/network problems, or API-level responses indicating invalid requests,
//! rate limits, and so on.
//!
//! # Examples
//!
//! ```rust
//! use chat_gpt_lib_rs::OpenAIError;
//!
//! fn example() -> Result<(), OpenAIError> {
//!     // Simulate an error scenario:
//!     Err(OpenAIError::ConfigError("No API key set".into()))
//! }
//! ```

use thiserror::Error;

/// Represents any error that can occur while using the OpenAI Rust client library.
///
/// This enum covers:
/// 1. Configuration errors, such as missing API keys or invalid builder settings.
/// 2. Network/HTTP errors encountered while making requests.
/// 3. Errors returned by the OpenAI API itself (e.g., rate limits, invalid parameters).
/// 4. JSON parsing errors (e.g., unexpected response formats).
#[derive(Debug, Error)]
pub enum OpenAIError {
    /// Errors related to invalid configuration or missing environment variables.
    #[error("Configuration Error: {0}")]
    ConfigError(String),

    /// Errors returned by the underlying HTTP client (`reqwest`).
    ///
    /// This typically indicates network-level issues, timeouts, TLS errors, etc.
    #[error("HTTP Error: {0}")]
    HTTPError(#[from] reqwest::Error),

    /// Errors that happen due to invalid or unexpected responses from the OpenAI API.
    ///
    /// For instance, if the response body is not valid JSON or doesn't match the expected schema.
    #[error("Deserialization/Parsing Error: {0}")]
    DeserializeError(#[from] serde_json::Error),

    /// Errors reported by the OpenAI API in its response body.
    ///
    /// This might include invalid request parameters, rate-limit violations, or internal
    /// server errors. The attached string typically contains a more descriptive message
    /// returned by the API.
    #[error("OpenAI API Error: {message}")]
    APIError {
        /// A short summary of what went wrong (as provided by the OpenAI API).
        message: String,
        /// The type/category of error (e.g. 'invalid_request_error', 'rate_limit_error', etc.).
        #[allow(dead_code)]
        err_type: Option<String>,
        /// An optional error code that might be returned by the OpenAI API.
        #[allow(dead_code)]
        code: Option<String>,
    },
}

impl OpenAIError {
    /// Creates an [`OpenAIError::APIError`] from detailed information about the error.
    ///
    /// # Parameters
    ///
    /// * `message` - A short description of the error.
    /// * `err_type` - The error type from OpenAI (e.g., "invalid_request_error").
    /// * `code` - An optional error code from OpenAI.
    ///
    /// # Example
    ///
    /// ```rust
    /// use chat_gpt_lib_rs::OpenAIError;
    ///
    /// let api_err = OpenAIError::api_error("Invalid request", Some("invalid_request_error"), None);
    /// ```
    pub fn api_error(
        message: impl Into<String>,
        err_type: Option<&str>,
        code: Option<&str>,
    ) -> Self {
        OpenAIError::APIError {
            message: message.into(),
            err_type: err_type.map(|s| s.to_string()),
            code: code.map(|s| s.to_string()),
        }
    }
}

/// An internal struct that represents the standard error response from the OpenAI API.
///
/// When the OpenAI API returns an error (e.g., 4xx or 5xx status code), it often includes
/// a JSON body describing the error. This struct captures those fields. Your code can
/// deserialize this into an [`OpenAIAPIErrorBody`] and then map it to an [`OpenAIError::APIError`].
#[derive(Debug, serde::Deserialize)]
pub(crate) struct OpenAIAPIErrorBody {
    /// The actual error details in a nested structure.
    pub error: OpenAIAPIErrorDetails,
}

/// The nested structure holding the error details returned by OpenAI.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct OpenAIAPIErrorDetails {
    /// A human-readable error message.
    pub message: String,
    /// The type/category of the error (e.g., "invalid_request_error", "rate_limit_error").
    #[serde(rename = "type")]
    pub err_type: String,
    /// An optional error code (e.g., "invalid_api_key").
    pub code: Option<String>,
}

impl From<OpenAIAPIErrorBody> for OpenAIError {
    fn from(body: OpenAIAPIErrorBody) -> Self {
        OpenAIError::APIError {
            message: body.error.message,
            err_type: Some(body.error.err_type),
            code: body.error.code,
        }
    }
}
