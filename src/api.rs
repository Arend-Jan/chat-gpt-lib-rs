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

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::config::OpenAIClient;
use crate::error::OpenAIError;

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
        // 1) Read raw text from the response
        let text = response.text().await?;

        // 2) Attempt to parse with serde_json. If it fails, map to `OpenAIError::DeserializeError`
        let parsed: R = serde_json::from_str(&text).map_err(OpenAIError::from)?; // This uses your `From<serde_json::Error> for OpenAIError::DeserializeError`

        Ok(parsed)
    } else {
        parse_error_response(response).await
    }
}

/// Attempts to parse the OpenAI error body. If successful, returns `Err(OpenAIError::APIError)`.
/// Otherwise, returns a generic error based on the HTTP status code or raw text.
pub async fn parse_error_response<R>(response: reqwest::Response) -> Result<R, OpenAIError> {
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

#[cfg(test)]
mod tests {
    /// # Tests for the `api` module
    ///
    /// These tests use [`wiremock`](https://crates.io/crates/wiremock) to **mock** HTTP responses from
    /// the OpenAI API, ensuring we can verify request-building, JSON handling, and error parsing logic
    /// without hitting real servers.
    ///
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::{OpenAIError, OpenAIError::APIError};
    use serde::Deserialize;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[derive(Debug, Deserialize)]
    struct MockResponse {
        pub foo: String,
        pub bar: i32,
    }

    /// Tests that `post_json` correctly sends a JSON POST request and parses a successful JSON response.
    #[tokio::test]
    async fn test_post_json_success() {
        // Start a local mock server
        let mock_server = MockServer::start().await;

        // Define an expected JSON response
        let mock_data = serde_json::json!({ "foo": "hello", "bar": 42 });

        // Mock a 200 OK response from the endpoint
        Mock::given(method("POST"))
            .and(path("/test-endpoint"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_data))
            .mount(&mock_server)
            .await;

        // Construct an OpenAIClient that points to our mock server URL
        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri()) // wiremock server
            .build()
            .unwrap();

        // We’ll send some dummy request body
        let request_body = serde_json::json!({ "dummy": true });

        // Call the function under test
        let result: Result<MockResponse, OpenAIError> =
            post_json(&client, "test-endpoint", &request_body).await;

        // Verify we got a success
        assert!(result.is_ok(), "Expected Ok, got Err");
        let parsed = result.unwrap();
        assert_eq!(parsed.foo, "hello");
        assert_eq!(parsed.bar, 42);
    }

    /// Tests that `post_json` handles non-2xx status codes and returns an `APIError`.
    #[tokio::test]
    async fn test_post_json_api_error() {
        let mock_server = MockServer::start().await;

        // Suppose the server returns a 400 with a JSON error body
        let error_body = serde_json::json!({
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "param": null,
                "code": "some_code"
            }
        });

        Mock::given(method("POST"))
            .and(path("/test-endpoint"))
            .respond_with(ResponseTemplate::new(400).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let request_body = serde_json::json!({ "dummy": true });

        let result: Result<MockResponse, OpenAIError> =
            post_json(&client, "test-endpoint", &request_body).await;

        // We should get an APIError with the parsed message
        match result {
            Err(APIError { message, .. }) => {
                assert!(
                    message.contains("Invalid request"),
                    "Expected error message about invalid request, got: {}",
                    message
                );
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    /// Tests that `post_json` surfaces a deserialization error if the server returns malformed JSON.
    #[tokio::test]
    async fn test_post_json_deserialize_error() {
        let mock_server = MockServer::start().await;

        // Return invalid JSON that won't match `MockResponse`
        let invalid_json = r#"{"foo": 123, "bar": "not_an_integer"}"#;

        Mock::given(method("POST"))
            .and(path("/test-endpoint"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(invalid_json, "application/json"))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let request_body = serde_json::json!({ "dummy": true });

        let result: Result<MockResponse, OpenAIError> =
            post_json(&client, "test-endpoint", &request_body).await;

        // We expect a DeserializeError
        assert!(matches!(result, Err(OpenAIError::DeserializeError(_))));
    }

    /// Tests that `get_json` properly sends a GET request and parses a successful JSON response.
    #[tokio::test]
    async fn test_get_json_success() {
        let mock_server = MockServer::start().await;

        let mock_data = serde_json::json!({ "foo": "abc", "bar": 99 });

        // Mock a GET response
        Mock::given(method("GET"))
            .and(path("/test-get"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_data))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Call the function under test
        let result: Result<MockResponse, OpenAIError> = get_json(&client, "test-get").await;

        // Check the result
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.foo, "abc");
        assert_eq!(parsed.bar, 99);
    }

    /// Tests that `get_json` handles a non-successful status code with an error body.
    #[tokio::test]
    async fn test_get_json_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = serde_json::json!({
            "error": {
                "message": "Resource not found",
                "type": "not_found",
                "code": "missing_resource"
            }
        });

        Mock::given(method("GET"))
            .and(path("/test-get"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result: Result<MockResponse, OpenAIError> = get_json(&client, "test-get").await;

        match result {
            Err(APIError { message, .. }) => {
                assert!(message.contains("Resource not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }
}
