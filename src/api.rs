//! The `api` module contains low-level functions for making HTTP requests to the OpenAI API.
//! It handles authentication headers, organization headers, error parsing, and JSON (de)serialization.
//!
//! # Usage
//!
//! This module is not typically used directly. Instead, higher-level modules (e.g., for
//! Completions, Chat, Embeddings, etc.) will call these functions to perform network requests.
//!
//! ```ignore
//! // Example usage in a hypothetical endpoints module:
//! use crate::api::{post_json};
//! use crate::config::OpenAIClient;
//! use crate::error::OpenAIError;
//!
//! // Suppose you have a function that creates a text completion...
//! pub async fn create_completion(
//!     client: &OpenAIClient,
//!     prompt: &str
//! ) -> Result<CompletionResponse, OpenAIError> {
//!     let request_body = CompletionRequest {
//!         model: "text-davinci-003".to_owned(),
//!         prompt: prompt.to_owned(),
//!         max_tokens: 100,
//!         temperature: 0.7,
//!     };
//!
//!     post_json(client, "completions", &request_body).await
//! }
//! ```

use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::config::OpenAIClient;
use crate::error::{OpenAIAPIErrorBody, OpenAIError};

/// Sends a POST request with a JSON body to the given `endpoint`.
///
/// # Parameters
///
/// - `client`: The [`OpenAIClient`](crate::config::OpenAIClient) holding base URL, API key, and a configured `reqwest::Client`.
/// - `endpoint`: The relative path (e.g. `"completions"`) appended to the base URL.
/// - `body`: A serializable request body (e.g. your request struct).
///
/// # Returns
///
/// A `Result` containing the response deserialized into type `R` on success, or an [`OpenAIError`]
/// on failure (e.g. network, JSON parse, or API error).
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: If the network request fails (e.g. timeout, DNS error).
/// - [`OpenAIError::DeserializeError`]: If the response JSON can’t be parsed into `R`.
/// - [`OpenAIError::APIError`]: If the OpenAI API indicates an error in the response body (e.g. invalid request).
pub(crate) async fn post_json<T, R>(
    client: &OpenAIClient,
    endpoint: &str,
    body: &T,
) -> Result<R, OpenAIError>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);
    let mut request_builder = client.http_client.post(&url).bearer_auth(client.api_key());

    // If an organization ID is configured, include that in the request headers.
    if let Some(org_id) = client.organization() {
        request_builder = request_builder.header("OpenAI-Organization", org_id);
    }

    let response = request_builder.json(body).send().await?;

    handle_response(response).await
}

/// Sends a GET request to the given `endpoint`.
///
/// # Parameters
///
/// - `client`: The [`OpenAIClient`](crate::config::OpenAIClient) holding base URL, API key, and a configured `reqwest::Client`.
/// - `endpoint`: The relative path (e.g. `"models"`) appended to the base URL.
///
/// # Returns
///
/// A `Result` containing the response deserialized into type `R` on success, or an [`OpenAIError`]
/// on failure (e.g. network, JSON parse, or API error).
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: If the network request fails (e.g. timeout, DNS error).
/// - [`OpenAIError::DeserializeError`]: If the response JSON can’t be parsed into `R`.
/// - [`OpenAIError::APIError`]: If the OpenAI API indicates an error in the response body (e.g. invalid request).
pub(crate) async fn get_json<R>(client: &OpenAIClient, endpoint: &str) -> Result<R, OpenAIError>
where
    R: DeserializeOwned,
{
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);
    let mut request_builder = client.http_client.get(&url).bearer_auth(client.api_key());

    // If an organization ID is configured, include that in the request headers.
    if let Some(org_id) = client.organization() {
        request_builder = request_builder.header("OpenAI-Organization", org_id);
    }

    let response = request_builder.send().await?;

    handle_response(response).await
}

/// Parses the `reqwest::Response` from the OpenAI API, returning a successful `R` or an
/// [`OpenAIError`].
///
/// # Parameters
///
/// - `response`: The raw HTTP response from `reqwest`.
///
/// # Returns
///
/// * `Ok(R)` if the response is `2xx` and can be deserialized into `R`.
/// * `Err(OpenAIError::APIError)` if the response has a non-success status code and includes
///   an OpenAI error message.
/// * `Err(OpenAIError::DeserializeError)` if the JSON could not be deserialized into `R`.
async fn handle_response<R>(response: reqwest::Response) -> Result<R, OpenAIError>
where
    R: DeserializeOwned,
{
    let status = response.status();
    if status.is_success() {
        // Deserialize the success response.
        let parsed_response = response.json::<R>().await?;
        Ok(parsed_response)
    } else {
        // Attempt to parse the error body returned by the OpenAI API.
        parse_error_response(response).await
    }
}

/// Attempts to parse the OpenAI error body. If successful, returns `Err(OpenAIError::APIError)`.
/// Otherwise, returns a generic error based on the HTTP status code or raw text.
async fn parse_error_response<R>(response: reqwest::Response) -> Result<R, OpenAIError> {
    let status = response.status();
    let text_body = response.text().await.unwrap_or_else(|_| "".to_string());

    match serde_json::from_str::<crate::error::OpenAIAPIErrorBody>(&text_body) {
        Ok(body) => Err(OpenAIError::from(body)), // Convert to OpenAIError::APIError
        Err(_) => {
            // If we couldn't parse the expected error JSON, return a more generic error message.
            let msg = format!(
                "HTTP {} returned from OpenAI API; body: {}",
                status, text_body
            );
            Err(OpenAIError::APIError {
                message: msg,
                err_type: None,
                code: None,
            })
        }
    }
}
