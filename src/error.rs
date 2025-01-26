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

#[cfg(test)]
mod tests {
    //! # Tests for the `error` module
    //!
    //! We verify each variant of [`OpenAIError`] along with its convenience methods
    //! (e.g. `api_error`) and `From<OpenAIAPIErrorBody>` implementation. This ensures
    //! that errors are created, displayed, and converted as expected under various conditions.

    use super::*;
    use std::fmt::Write as _;

    /// Produces a `reqwest::Error` by making a **blocking** request to an invalid URL.
    /// This requires `reqwest` with the `"blocking"` feature enabled.
    fn produce_reqwest_error() -> reqwest::Error {
        // Attempting to make a request to a non-routable domain or an invalid protocol
        reqwest::blocking::Client::new()
            .get("http://this-domain-should-not-exist9999.test")
            .send()
            .unwrap_err()
    }

    /// Produces a `serde_json::Error` by parsing invalid JSON.
    fn produce_serde_json_error() -> serde_json::Error {
        serde_json::from_str::<serde_json::Value>("\"unterminated string").unwrap_err()
    }

    #[test]
    fn test_config_error() {
        let err = OpenAIError::ConfigError("No API key found".to_string());
        let display_str = format!("{}", err);

        // Verify it's the correct variant
        match &err {
            OpenAIError::ConfigError(msg) => {
                assert_eq!(msg, "No API key found");
            }
            other => panic!("Expected ConfigError, got: {:?}", other),
        }

        // Check Display output
        assert!(
            display_str.contains("No API key found"),
            "Display should contain the config error message, got: {}",
            display_str
        );
    }

    #[test]
    fn test_http_error() {
        let reqwest_err = produce_reqwest_error();
        let err = OpenAIError::HTTPError(reqwest_err);

        let display_str = format!("{}", err);
        assert!(
            display_str.contains("HTTP Error:"),
            "Should contain 'HTTP Error:' prefix, got: {}",
            display_str
        );

        // Pattern-match on &err
        match &err {
            OpenAIError::HTTPError(e) => {
                let e_str = format!("{}", e);
                // Accept multiple possible error messages
                assert!(
                    e_str.contains("error sending request")
                        || e_str.contains("dns error")
                        || e_str.contains("Could not resolve host")
                        || e_str.contains("Name or service not known"),
                    "Expected mention of DNS/resolve error or sending request, got: {}",
                    e_str
                );
            }
            other => panic!("Expected HTTPError, got: {:?}", other),
        }
    }

    #[test]
    fn test_deserialize_error() {
        // Produce a serde_json::Error, then convert to OpenAIError
        let serde_err = produce_serde_json_error();
        let err = OpenAIError::DeserializeError(serde_err);

        // Display
        let display_str = format!("{}", err);
        assert!(
            display_str.contains("Deserialization/Parsing Error:"),
            "Should contain 'Deserialization/Parsing Error:', got: {}",
            display_str
        );

        // Pattern-match on &err to avoid partial moves
        match &err {
            OpenAIError::DeserializeError(e) => {
                let e_str = format!("{}", e);
                assert!(
                    e_str.contains("EOF while parsing a string")
                        || e_str.contains("unterminated string"),
                    "Expected mention of parse error about unterminated, got: {}",
                    e_str
                );
            }
            other => panic!("Expected DeserializeError, got: {:?}", other),
        }
    }

    #[test]
    fn test_api_error() {
        // Create an APIError variant via the convenience method
        let err = OpenAIError::api_error(
            "Something went wrong",
            Some("invalid_request_error"),
            Some("ERR123"),
        );
        let display_str = format!("{}", err);

        match &err {
            OpenAIError::APIError {
                message,
                err_type,
                code,
            } => {
                assert_eq!(message, "Something went wrong");
                assert_eq!(err_type.as_deref(), Some("invalid_request_error"));
                assert_eq!(code.as_deref(), Some("ERR123"));
            }
            other => panic!("Expected APIError, got: {:?}", other),
        }

        // Check Display output
        assert!(
            display_str.contains("OpenAI API Error: Something went wrong"),
            "Expected 'OpenAI API Error:' prefix, got: {}",
            display_str
        );
    }

    #[test]
    fn test_from_openaiapierrorbody() {
        let body = OpenAIAPIErrorBody {
            error: OpenAIAPIErrorDetails {
                message: "Rate limit exceeded".to_string(),
                err_type: "rate_limit_error".to_string(),
                code: Some("rate_limit_code".to_string()),
            },
        };
        let err = OpenAIError::from(body);

        match &err {
            OpenAIError::APIError {
                message,
                err_type,
                code,
            } => {
                assert_eq!(message, "Rate limit exceeded");
                assert_eq!(err_type.as_deref(), Some("rate_limit_error"));
                assert_eq!(code.as_deref(), Some("rate_limit_code"));
            }
            other => panic!("Expected APIError from error body, got: {:?}", other),
        }
    }

    #[test]
    fn test_display_trait_all_variants() {
        let config_err = OpenAIError::ConfigError("missing key".to_string());
        let http_err = OpenAIError::HTTPError(produce_reqwest_error());
        let deser_err = OpenAIError::DeserializeError(produce_serde_json_error());
        let api_err = OpenAIError::api_error("Remote server said no", Some("some_api_error"), None);

        let mut combined = String::new();
        writeln!(&mut combined, "{}", config_err).unwrap();
        writeln!(&mut combined, "{}", http_err).unwrap();
        writeln!(&mut combined, "{}", deser_err).unwrap();
        writeln!(&mut combined, "{}", api_err).unwrap();

        // Just a quick check of the combined output
        assert!(combined.contains("Configuration Error: missing key"));
        assert!(combined.contains("HTTP Error:"));
        assert!(combined.contains("Deserialization/Parsing Error:"));
        assert!(combined.contains("OpenAI API Error: Remote server said no"));
    }
}
