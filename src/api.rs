//! The `api` module contains low-level functions for making HTTP requests to the OpenAI API.
//! It handles authentication headers, organization headers, error parsing, and JSON (de)serialization.
//!
//! # Usage
//!
//! This module is not typically used directly. Instead, higher-level modules (e.g., for
//! Completions, Chat, Embeddings, etc.) will call these functions to perform network requests.

use crate::config::OpenAIClient;
use crate::error::OpenAIError;
use reqwest::Method;
use serde::Serialize;
use serde::de::DeserializeOwned;

// Import for streaming support:
use futures_util::stream::TryStreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_stream::Stream; // Trait for streams.
use tokio_stream::StreamExt as TokioStreamExt; // Needed for filter_map.
use tokio_stream::wrappers::LinesStream;
use tokio_util::io::StreamReader;

fn endpoint_url(client: &OpenAIClient, endpoint: &str) -> String {
    format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint)
}

fn with_common_headers(
    client: &OpenAIClient,
    request_builder: reqwest::RequestBuilder,
) -> reqwest::RequestBuilder {
    let mut request_builder = request_builder.bearer_auth(client.api_key());

    if let Some(org_id) = client.organization() {
        request_builder = request_builder.header("OpenAI-Organization", org_id);
    }

    request_builder
}

fn request_builder(
    client: &OpenAIClient,
    method: Method,
    endpoint: &str,
) -> reqwest::RequestBuilder {
    let url = endpoint_url(client, endpoint);
    with_common_headers(client, client.http_client.request(method, url))
}

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
    let response = request_builder(client, Method::POST, endpoint)
        .json(body)
        .send()
        .await?;

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
    let response = request_builder(client, Method::GET, endpoint)
        .send()
        .await?;

    handle_response(response).await
}

/// Sends a DELETE request to the given `endpoint` and deserializes a successful JSON response.
pub(crate) async fn delete_json<R>(client: &OpenAIClient, endpoint: &str) -> Result<R, OpenAIError>
where
    R: DeserializeOwned,
{
    let response = request_builder(client, Method::DELETE, endpoint)
        .send()
        .await?;
    handle_response(response).await
}

/// Sends a GET request to the given `endpoint` and returns the raw body as bytes.
pub(crate) async fn get_bytes(
    client: &OpenAIClient,
    endpoint: &str,
) -> Result<Vec<u8>, OpenAIError> {
    let response = request_builder(client, Method::GET, endpoint)
        .send()
        .await?;

    if response.status().is_success() {
        Ok(response.bytes().await?.to_vec())
    } else {
        parse_error_response(response).await
    }
}

/// Sends a POST request with multipart form data to the given `endpoint`.
pub(crate) async fn post_multipart<R>(
    client: &OpenAIClient,
    endpoint: &str,
    form: reqwest::multipart::Form,
) -> Result<R, OpenAIError>
where
    R: DeserializeOwned,
{
    let response = request_builder(client, Method::POST, endpoint)
        .multipart(form)
        .send()
        .await?;

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
        let parsed: R = serde_json::from_str(&text).map_err(OpenAIError::from)?;

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
        Ok(body) => Err(OpenAIError::from(body)),
        Err(_) => {
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

/// Sends a POST request with a JSON body to the given `endpoint` and returns a stream of responses.
/// This is designed for endpoints that support streaming responses (e.g., Chat Completions with `stream = true`).
///
/// # Parameters
///
/// - `client`: The [`OpenAIClient`] holding base URL, API key, etc.
/// - `endpoint`: The relative endpoint (e.g., `"chat/completions"`) appended to the base URL.
/// - `body`: A serializable request body.
///
/// # Returns
///
/// A stream of deserialized items of type `R`. Each item represents a partial response from the server.
///
/// # Errors
///
/// Returns an [`OpenAIError`] if the initial request fails or if the HTTP response indicates an error.
///
/// # Dependencies
///
/// This function uses the latest versions of `tokio-stream` and `tokio-util`.
pub async fn post_json_stream<T, R>(
    client: &OpenAIClient,
    endpoint: &str,
    body: &T,
) -> Result<impl Stream<Item = Result<R, OpenAIError>>, OpenAIError>
where
    T: Serialize,
    R: DeserializeOwned + 'static,
{
    let response = request_builder(client, Method::POST, endpoint)
        .json(body)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let text_body = response.text().await.unwrap_or_else(|_| "".to_string());
        match serde_json::from_str::<crate::error::OpenAIAPIErrorBody>(&text_body) {
            Ok(body_err) => return Err(OpenAIError::from(body_err)),
            Err(_) => {
                return Err(OpenAIError::APIError {
                    message: format!(
                        "HTTP {} returned from OpenAI API; body: {}",
                        status, text_body
                    ),
                    err_type: None,
                    code: None,
                });
            }
        }
    }

    // Convert the response's byte stream into an async reader.
    let byte_stream = response.bytes_stream().map_err(std::io::Error::other);
    let stream_reader = StreamReader::new(byte_stream);
    let buf_reader = BufReader::new(stream_reader);

    // Create a stream of lines from the buffered reader.
    let lines = LinesStream::new(buf_reader.lines());

    // Process each line synchronously:
    //   - Ignore empty lines or those that contain "[DONE]".
    //   - Remove the "data:" prefix if present.
    //   - Attempt to deserialize the remaining JSON into type `R`.
    let stream = lines.filter_map(|line_result| {
        match line_result {
            Ok(line) => {
                let trimmed = line.trim();
                // Skip empty lines or termination markers.
                if trimmed.is_empty() || trimmed.contains("[DONE]") {
                    None
                } else {
                    // Remove the "data:" prefix if it exists.
                    let data = if trimmed.starts_with("data:") {
                        trimmed.trim_start_matches("data:").trim()
                    } else {
                        trimmed
                    };
                    // Attempt to deserialize the JSON.
                    match serde_json::from_str::<R>(data) {
                        Ok(parsed) => Some(Ok(parsed)),
                        Err(e) => Some(Err(OpenAIError::from(e))),
                    }
                }
            }
            Err(e) => Some(Err(OpenAIError::from(e))),
        }
    });
    Ok(stream)
}

#[cfg(test)]
mod tests {
    /// # Tests for the `api` module
    ///
    /// These tests use [`wiremock`](https://crates.io/crates/wiremock) to **mock** HTTP responses from
    /// the OpenAI API, ensuring we can verify request-building, JSON handling, and error parsing logic
    /// without hitting real servers.
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::{OpenAIError, OpenAIError::APIError};
    use serde::Deserialize;
    use tokio_stream::StreamExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate}; // <-- for `next`, `collect`, etc.

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
            .with_base_url(&mock_server.uri())
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

    /* ---------------------------------------------------------------- *\
     *  post_json_stream – successful Server-Sent Events (SSE) parsing *
    \* ---------------------------------------------------------------- */
    #[tokio::test]
    async fn test_post_json_stream_success() {
        let mock_server = MockServer::start().await;

        // Simulated event-stream (empty line, "data:" prefix, [DONE] marker)
        let body = r#"data: {"foo":"first","bar":1}
data: {"foo":"second","bar":2}
data: [DONE]
"#;

        Mock::given(method("POST"))
            .and(path("/stream-endpoint"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Any serialisable body works for this test.
        let req_body = serde_json::json!({ "stream": true });

        let stream = post_json_stream::<serde_json::Value, MockResponse>(
            &client,
            "stream-endpoint",
            &req_body,
        )
        .await
        .expect("stream should start OK");

        // Collect all chunks, unwrapping individual `Result`s.
        let items: Vec<MockResponse> = stream
            .map(|r| r.expect("chunk should be Ok"))
            .collect()
            .await;

        assert_eq!(items.len(), 2);
        assert_eq!(items[0].foo, "first");
        assert_eq!(items[0].bar, 1);
        assert_eq!(items[1].foo, "second");
        assert_eq!(items[1].bar, 2);
    }

    #[tokio::test]
    async fn test_post_json_stream_emits_deserialize_error_chunk() {
        let mock_server = MockServer::start().await;

        let body = r#"data: {"foo":"first","bar":1}
data: {"foo":"oops","bar":"not-a-number"}
data: {"foo":"second","bar":2}
data: [DONE]
"#;

        Mock::given(method("POST"))
            .and(path("/stream-endpoint"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/event-stream"))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req_body = serde_json::json!({ "stream": true });

        let mut stream = post_json_stream::<serde_json::Value, MockResponse>(
            &client,
            "stream-endpoint",
            &req_body,
        )
        .await
        .expect("stream should start OK");

        let mut ok_chunks = Vec::new();
        let mut saw_deserialize_error = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => ok_chunks.push(chunk.foo),
                Err(OpenAIError::DeserializeError(_)) => saw_deserialize_error = true,
                Err(other) => panic!("Expected deserialize error, got {other:?}"),
            }
        }

        assert_eq!(ok_chunks, vec!["first".to_string(), "second".to_string()]);
        assert!(saw_deserialize_error);
    }

    /* -------------------------------------------------------------- *\
     * post_json_stream – HTTP error returns an OpenAI APIError     *
    \* -------------------------------------------------------------- */
    #[tokio::test]
    async fn test_post_json_stream_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = serde_json::json!({
            "error": {
                "message": "Streaming not allowed",
                "type": "invalid_request_error",
                "code": "not_allowed"
            }
        });

        Mock::given(method("POST"))
            .and(path("/stream-endpoint"))
            .respond_with(ResponseTemplate::new(403).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req_body = serde_json::json!({ "stream": true });

        let result = post_json_stream::<serde_json::Value, MockResponse>(
            &client,
            "stream-endpoint",
            &req_body,
        )
        .await;

        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Streaming not allowed"));
            }
            Ok(_) => panic!("Expected APIError, got Ok(_)"),
            Err(other) => panic!("Expected APIError, got {:?}", other.to_string()),
        }
    }

    /* ----------------------------------------------------------------- *\
     * parse_error_response – fallback branch with plain-text body   *
    \* ----------------------------------------------------------------- */

    #[tokio::test]
    async fn test_post_json_plain_text_error_fallback() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/plain-error"))
            .respond_with(
                ResponseTemplate::new(503).set_body_raw("Service unavailable", "text/plain"),
            )
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req_body = serde_json::json!({});

        let result: Result<MockResponse, OpenAIError> =
            post_json(&client, "plain-error", &req_body).await;

        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(
                    message.contains("Service unavailable"),
                    "Unexpected message: {message}"
                );
            }
            other => panic!("Expected generic APIError, got {:?}", other),
        }
    }
}
