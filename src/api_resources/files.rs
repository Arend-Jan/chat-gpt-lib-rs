//! This module provides functionality for working with files using the
//! [OpenAI Files API](https://platform.openai.com/docs/api-reference/files).
//!
//! Typical usage includes uploading a JSONL file for fine-tuning or other purposes,
//! listing all files, retrieving file metadata, deleting a file, or even downloading
//! its contents.
//!
//! # Workflow
//!
//! 1. **Upload a file** with [`upload_file`] (usually a `.jsonl` file for fine-tuning data).
//! 2. **List files** with [`list_files`], which returns metadata for all uploaded files.
//! 3. **Retrieve file metadata** with [`retrieve_file_metadata`] for a specific file ID.
//! 4. **Delete a file** you no longer need with [`delete_file`].
//! 5. **Download file content** with [`retrieve_file_content`], if necessary for debugging or reuse.
//!
//! # Example
//!
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::files::{upload_file, UploadFilePurpose};
//! use chat_gpt_lib_rs::OpenAIClient;
//! use chat_gpt_lib_rs::error::OpenAIError;
//! use std::path::PathBuf;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?;
//!
//!     // Suppose you have a JSONL file at "./training_data.jsonl" for fine-tuning
//!     let file_path = PathBuf::from("./training_data.jsonl");
//!
//!     // Upload the file with purpose "fine-tune"
//!     let file_obj = upload_file(&client, &file_path, UploadFilePurpose::FineTune).await?;
//!     println!("Uploaded file ID: {}", file_obj.id);
//!
//!     Ok(())
//! }
//! ```

use std::path::Path;

use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};

use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// The "purpose" parameter you must supply when uploading a file.
///
/// For fine-tuning, use `UploadFilePurpose::FineTune`.
/// For other potential upload workflows, consult the OpenAI docs.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum UploadFilePurpose {
    /// Indicates that this file will be used for fine-tuning.
    FineTune,
    /// If you want to specify another purpose (not commonly used).
    Other(String),
}

impl std::fmt::Display for UploadFilePurpose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UploadFilePurpose::FineTune => write!(f, "fine-tune"),
            UploadFilePurpose::Other(val) => write!(f, "{val}"),
        }
    }
}

/// Represents a file object in OpenAI.
///
/// For example, when you upload a file via `POST /v1/files`, the API responds with
/// this structure containing metadata about the file.
#[derive(Debug, Deserialize)]
pub struct FileObject {
    /// The ID of the file, e.g. "file-abc123".
    pub id: String,
    /// The object type, usually "file".
    pub object: String,
    /// The size of the file in bytes.
    pub bytes: u64,
    /// The time (in epoch seconds) when the file was uploaded.
    pub created_at: u64,
    /// The filename you provided during upload.
    pub filename: String,
    /// The purpose for which the file was uploaded (e.g. "fine-tune").
    pub purpose: String,
    /// The current status of the file, e.g. "uploaded".
    pub status: Option<String>,
    /// More detailed status information, if available.
    pub status_details: Option<String>,
}

/// A response type for listing files. Contains an array of [`FileObject`].
#[derive(Debug, Deserialize)]
pub struct FileListResponse {
    /// The object type, typically "list".
    pub object: String,
    /// The actual list of files.
    pub data: Vec<FileObject>,
}

/// Represents the response returned by the Delete File endpoint.
#[derive(Debug, Deserialize)]
pub struct DeleteFileResponse {
    /// The file ID that was deleted.
    pub id: String,
    /// The object type, usually "file".
    pub object: String,
    /// Indicates that the file was deleted.
    pub deleted: bool,
}

/// Uploads a file to OpenAI.
///
/// This requires multipart form data:
/// - A "file" field with the actual file bytes
/// - A "purpose" field with the reason for upload (e.g. "fine-tune")
///
/// The purpose is required by the API.
///
/// # Parameters
/// * `client` - The OpenAI client.
/// * `file_path` - Path to the local file to upload.
/// * `purpose` - The file's intended usage (e.g. `UploadFilePurpose::FineTune`).
///
/// # Returns
/// A [`FileObject`] containing metadata about the newly uploaded file.
///
/// # Errors
/// Returns [`OpenAIError`] if the network request fails, the file can’t be read,
/// or the API returns an error.
pub async fn upload_file(
    client: &OpenAIClient,
    file_path: &Path,
    purpose: UploadFilePurpose,
) -> Result<FileObject, OpenAIError> {
    let endpoint = "files";
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    // Prepare the multipart form
    let file_bytes = tokio::fs::read(file_path)
        .await
        .map_err(|e| OpenAIError::ConfigError(format!("Failed to read file: {}", e)))?;
    let filename = file_path
        .file_name()
        .map(|os| os.to_string_lossy().into_owned())
        .unwrap_or_else(|| "upload.bin".to_string());

    let file_part = Part::bytes(file_bytes)
        .file_name(filename)
        .mime_str("application/octet-stream")
        .unwrap_or_else(|_| {
            // In a real scenario, if mime_str fails, we fallback to a default
            Part::bytes(Vec::new()).file_name("default.bin")
        });

    // The "purpose" must be a string field in the form
    let form = Form::new()
        .part("file", file_part)
        .text("purpose", purpose.to_string());

    // Send the request
    let response = client
        .http_client
        .post(&url)
        .bearer_auth(client.api_key())
        .multipart(form)
        .send()
        .await?;

    handle_file_response(response).await
}

/// Lists all files stored in your OpenAI account.
///
/// # Returns
/// A [`FileListResponse`] containing metadata for each file.
///
/// # Errors
/// Returns [`OpenAIError`] if the request fails or the API returns an error.
pub async fn list_files(client: &OpenAIClient) -> Result<FileListResponse, OpenAIError> {
    let endpoint = "files";
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    let response = client
        .http_client
        .get(&url)
        .bearer_auth(client.api_key())
        .send()
        .await?;

    let status = response.status();
    if status.is_success() {
        let files = response.json::<FileListResponse>().await?;
        Ok(files)
    } else {
        crate::api::parse_error_response(response).await
    }
}

/// Retrieves metadata about a specific file by its ID.
///
/// # Parameters
/// * `file_id` - The file ID, e.g. "file-abc123"
///
/// # Returns
/// A [`FileObject`] representing the file's metadata.
pub async fn retrieve_file_metadata(
    client: &OpenAIClient,
    file_id: &str,
) -> Result<FileObject, OpenAIError> {
    let endpoint = format!("files/{}", file_id);
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    let response = client
        .http_client
        .get(&url)
        .bearer_auth(client.api_key())
        .send()
        .await?;

    handle_file_response(response).await
}

/// Downloads the content of a file by its ID.
///
/// **Note**: For fine-tuning `.jsonl` files, you can retrieve the training data
/// to verify or reuse it.
///
/// # Parameters
/// * `file_id` - The file ID to download.
///
/// # Returns
/// A `Vec<u8>` containing the raw file data.
pub async fn retrieve_file_content(
    client: &OpenAIClient,
    file_id: &str,
) -> Result<Vec<u8>, OpenAIError> {
    // The official docs:
    // GET /v1/files/{file_id}/content
    let endpoint = format!("files/{}/content", file_id);
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    let response = client
        .http_client
        .get(&url)
        .bearer_auth(client.api_key())
        .send()
        .await?;

    if response.status().is_success() {
        let bytes = response.bytes().await?;
        Ok(bytes.to_vec())
    } else {
        crate::api::parse_error_response(response).await
    }
}

/// Deletes a file by its ID.
///
/// # Parameters
/// * `file_id` - The file ID, e.g. "file-abc123"
///
/// # Returns
/// A [`DeleteFileResponse`] indicating success or failure (the `deleted` field
/// should be true if it succeeds).
pub async fn delete_file(
    client: &OpenAIClient,
    file_id: &str,
) -> Result<DeleteFileResponse, OpenAIError> {
    let endpoint = format!("files/{}", file_id);
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    let response = client
        .http_client
        .delete(&url)
        .bearer_auth(client.api_key())
        .send()
        .await?;

    let status = response.status();
    if status.is_success() {
        let res_body = response.json::<DeleteFileResponse>().await?;
        Ok(res_body)
    } else {
        crate::api::parse_error_response(response).await
    }
}

/// Helper to handle responses that should yield a [`FileObject`].
async fn handle_file_response(response: reqwest::Response) -> Result<FileObject, OpenAIError> {
    let status = response.status();
    if status.is_success() {
        let file_obj = response.json::<FileObject>().await?;
        Ok(file_obj)
    } else {
        crate::api::parse_error_response(response).await
    }
}

#[cfg(test)]
mod tests {
    //! # Tests for the `files` module
    //!
    //! Here we use [`wiremock`](https://crates.io/crates/wiremock) to simulate OpenAI's Files API:
    //! 1. **Upload** a file with multipart form data
    //! 2. **List** files
    //! 3. **Retrieve** file metadata
    //! 4. **Delete** a file
    //! 5. **Retrieve** file content
    //!
    //! We test both **successful** (2xx) cases and **error** cases (non-2xx with an OpenAI-style error).
    //! For the file upload, we use a temporary in-memory file to emulate reading bytes.

    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use std::io::Write as _;
    use tempfile::NamedTempFile;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Creates a temporary file with specified contents for testing upload.
    /// Returns the `NamedTempFile` handle so it gets cleaned up automatically.
    fn create_temp_file(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("Failed to create temp file");
        write!(file, "{}", contents).expect("Unable to write to temp file");
        file
    }

    #[tokio::test]
    async fn test_upload_file_success() {
        // Start a local mock server
        let mock_server = MockServer::start().await;

        // Simulate a successful 200 JSON response with file metadata
        let success_body = json!({
            "id": "file-abc123",
            "object": "file",
            "bytes": 1024,
            "created_at": 1673643147,
            "filename": "mydata.jsonl",
            "purpose": "fine-tune",
            "status": "uploaded",
            "status_details": null
        });

        Mock::given(method("POST"))
            .and(path("/files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        // Create a test client
        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        // Create a temp file to mock reading local data
        let temp_file = create_temp_file("some jsonl contents");
        let result = upload_file(&client, temp_file.path(), UploadFilePurpose::FineTune).await;
        assert!(result.is_ok(), "Expected success, got: {:?}", result);

        let file_obj = result.unwrap();
        assert_eq!(file_obj.id, "file-abc123");
        assert_eq!(file_obj.object, "file");
        assert_eq!(file_obj.bytes, 1024);
        assert_eq!(file_obj.filename, "mydata.jsonl");
        assert_eq!(file_obj.purpose, "fine-tune");
        assert_eq!(file_obj.status.as_deref(), Some("uploaded"));
    }

    #[tokio::test]
    async fn test_upload_file_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "File size too large",
                "type": "invalid_request_error",
                "code": "file_size_exceeded"
            }
        });

        // Return a 400 for the file upload
        Mock::given(method("POST"))
            .and(path("/files"))
            .respond_with(ResponseTemplate::new(400).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let temp_file = create_temp_file("some jsonl contents");
        let result = upload_file(&client, temp_file.path(), UploadFilePurpose::FineTune).await;

        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("File size too large"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_upload_file_config_error_when_file_missing() {
        // Test reading a non-existent file, which triggers a ConfigError from `upload_file`.
        let mock_server = MockServer::start().await;
        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let non_existent_path = std::path::Path::new("/some/path/that/does/not/exist.jsonl");
        let result = upload_file(&client, non_existent_path, UploadFilePurpose::FineTune).await;
        match result {
            Err(OpenAIError::ConfigError(msg)) => {
                assert!(
                    msg.contains("Failed to read file:"),
                    "Expected a file read error, got: {}",
                    msg
                );
            }
            other => panic!("Expected ConfigError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_files_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "object": "list",
            "data": [
                {
                    "id": "file-xyz789",
                    "object": "file",
                    "bytes": 999,
                    "created_at": 1673644000,
                    "filename": "data.jsonl",
                    "purpose": "fine-tune",
                    "status": "uploaded",
                    "status_details": null
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path("/files"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_files(&client).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let files = result.unwrap();
        assert_eq!(files.object, "list");
        assert_eq!(files.data.len(), 1);
        let file_obj = &files.data[0];
        assert_eq!(file_obj.id, "file-xyz789");
        assert_eq!(file_obj.bytes, 999);
    }

    #[tokio::test]
    async fn test_list_files_api_error() {
        let mock_server = MockServer::start().await;
        let error_body = json!({
            "error": {
                "message": "Could not list files",
                "type": "internal_server_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path("/files"))
            .respond_with(ResponseTemplate::new(500).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_files(&client).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Could not list files"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retrieve_file_metadata_success() {
        let mock_server = MockServer::start().await;
        let success_body = json!({
            "id": "file-abc123",
            "object": "file",
            "bytes": 2048,
            "created_at": 1673645000,
            "filename": "info.jsonl",
            "purpose": "fine-tune",
            "status": "uploaded",
            "status_details": null
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/files/file-abc123$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_file_metadata(&client, "file-abc123").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let file_obj = result.unwrap();
        assert_eq!(file_obj.id, "file-abc123");
        assert_eq!(file_obj.bytes, 2048);
        assert_eq!(file_obj.filename, "info.jsonl");
    }

    #[tokio::test]
    async fn test_retrieve_file_metadata_api_error() {
        let mock_server = MockServer::start().await;
        let error_body = json!({
            "error": {
                "message": "File not found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/files/file-999$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_file_metadata(&client, "file-999").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("File not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retrieve_file_content_success() {
        let mock_server = MockServer::start().await;
        let file_data = b"this is the file content";

        Mock::given(method("GET"))
            .and(path_regex(r"^/files/file-abc123/content$"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(file_data, "application/octet-stream"),
            )
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_file_content(&client, "file-abc123").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let content = result.unwrap();
        assert_eq!(content, file_data);
    }

    #[tokio::test]
    async fn test_retrieve_file_content_api_error() {
        let mock_server = MockServer::start().await;
        let error_body = json!({
            "error": {
                "message": "Content not found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/files/file-000/content$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_file_content(&client, "file-000").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Content not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_delete_file_success() {
        let mock_server = MockServer::start().await;
        let success_body = json!({
            "id": "file-abc123",
            "object": "file",
            "deleted": true
        });

        Mock::given(method("DELETE"))
            .and(path_regex(r"^/files/file-abc123$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = delete_file(&client, "file-abc123").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let del_resp = result.unwrap();
        assert_eq!(del_resp.id, "file-abc123");
        assert_eq!(del_resp.object, "file");
        assert!(del_resp.deleted);
    }

    #[tokio::test]
    async fn test_delete_file_api_error() {
        let mock_server = MockServer::start().await;
        let error_body = json!({
            "error": {
                "message": "No file with ID file-xyz found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("DELETE"))
            .and(path_regex(r"^/files/file-xyz$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = delete_file(&client, "file-xyz").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("No file with ID file-xyz found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }
}
