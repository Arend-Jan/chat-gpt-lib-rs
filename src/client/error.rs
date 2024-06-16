use reqwest;
use thiserror::Error;

/// Enum representing possible errors in the ChatGPTClient.
///
/// Variants:
/// - `RequestFailed`: Indicates that the request failed with a specific message.
/// - `Reqwest`: Represents an error that occurred in the `reqwest` library.
#[derive(Debug, Error)]
pub enum ChatGPTError {
    #[error("Request failed with message: {0}")]
    RequestFailed(String),
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),
}
