//! This module provides functionality for working with fine-tuning jobs using the
//! [OpenAI Fine-tunes API](https://platform.openai.com/docs/api-reference/fine-tunes).
//!
//! Fine-tuning allows you to train a model on custom data so it can better handle
//! domain-specific terminology or style. You start by uploading training data as a
//! file, and optionally a validation file. Then you create a fine-tune job pointing
//! to those files. Once the job is finished, you can use the resulting fine-tuned
//! model for completions or other tasks.
//!
//! # Overview
//!
//! 1. **Upload training file** (outside the scope of this module, see the Files API).
//! 2. **Create a fine-tune job** with [`create_fine_tune`].
//! 3. **List fine-tunes** with [`list_fine_tunes`].
//! 4. **Retrieve a fine-tune** with [`retrieve_fine_tune`].
//! 5. **Cancel a fine-tune** with [`cancel_fine_tune`], if needed.
//! 6. **List fine-tune events** with [`list_fine_tune_events`] (to see training progress).
//! 7. **Delete fine-tuned model** with [`delete_fine_tune_model`], if you want to remove it.
//!
//! # Example
//! ```rust,no_run
//! use chat_gpt_lib_rs::api_resources::fine_tunes::{create_fine_tune, CreateFineTuneRequest};
//! use chat_gpt_lib_rs::error::OpenAIError;
//! use chat_gpt_lib_rs::OpenAIClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OpenAIError> {
//!     let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY
//!
//!     // Create a fine-tune job (assumes you've already uploaded a file and obtained its ID).
//!     let request = CreateFineTuneRequest {
//!         training_file: "file-abc123".to_string(),
//!         model: Some("curie".to_string()),
//!         ..Default::default()
//!     };
//!
//!     let response = create_fine_tune(&client, &request).await?;
//!     println!("Created fine-tune: {}", response.id);
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::api::{get_json, parse_error_response, post_json};
use crate::config::OpenAIClient;
use crate::error::OpenAIError;

/// A request struct for creating a fine-tune job.
///
/// Required parameter: `training_file` (the file ID of your training data).
///
/// Other fields are optional or have defaults. See [OpenAI Docs](https://platform.openai.com/docs/api-reference/fine-tunes/create)
/// for details on each parameter.
#[derive(Debug, Serialize, Default, Clone)]
pub struct CreateFineTuneRequest {
    /// The ID of an uploaded file that contains training data.
    ///
    /// See the Files API to upload a file and get this ID.  
    /// **Required**.
    pub training_file: String,

    /// The ID of an uploaded file that contains validation data.
    /// If `None`, no validation is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,

    /// The model to start fine-tuning from (e.g. "curie").  
    /// Defaults to "curie" if `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The number of epochs to train the model for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<u32>,

    /// The batch size to use for training.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,

    /// The learning rate multiplier to use.
    /// The fine-tune API will pick a default based on dataset size if `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f64>,

    /// The weight to assign to the prompt loss relative to the completion loss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_loss_weight: Option<f64>,

    /// If `true`, calculates classification-specific metrics such as accuracy
    /// and F-1, assuming the training data is intended for classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_classification_metrics: Option<bool>,

    /// The number of classes in a classification task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_n_classes: Option<u32>,

    /// The positive class in a binary classification task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_positive_class: Option<String>,

    /// If this is specified, calculates F-beta scores at the specified beta values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_betas: Option<Vec<f64>>,

    /// A string of up to 40 characters that will be added to your fine-tuned model name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

/// Represents a fine-tune job, either newly created or retrieved from the API.
#[derive(Debug, Deserialize)]
pub struct FineTune {
    /// The ID of the fine-tune job, e.g. "ft-XXXX".
    pub id: String,
    /// The object type, usually "fine-tune".
    pub object: String,
    /// The creation time in epoch seconds.
    pub created_at: u64,
    /// The time when training was last updated in epoch seconds.
    pub updated_at: u64,
    /// The base model used for fine-tuning.
    pub model: String,
    /// The name of the resulting fine-tuned model, if available.
    pub fine_tuned_model: Option<String>,
    /// The current status of the fine-tune job (e.g. "pending", "succeeded", "cancelled").
    pub status: String,
    /// A list of events describing updates to the fine-tune job (optional).
    #[serde(default)]
    pub events: Vec<FineTuneEvent>,
}

/// Represents a single event in a fine-tune job's lifecycle (e.g., job enqueued, model trained).
#[derive(Debug, Deserialize)]
pub struct FineTuneEvent {
    /// The object type, usually "fine-tune-event".
    pub object: String,
    /// The time in epoch seconds of this event.
    pub created_at: u64,
    /// The log message describing the event.
    pub level: String,
    /// The actual event message.
    pub message: String,
}

/// The response for listing fine-tunes: an object with `"data"` containing an array of [`FineTune`].
#[derive(Debug, Deserialize)]
pub struct FineTuneList {
    /// Typically "list".
    pub object: String,
    /// The actual array of fine-tune jobs.
    pub data: Vec<FineTune>,
}

/// Creates a fine-tune job.
///
/// # Parameters
///
/// * `client` - The [`OpenAIClient`](crate::config::OpenAIClient).
/// * `request` - The [`CreateFineTuneRequest`] with mandatory `training_file` and other optional fields.
///
/// # Returns
///
/// A [`FineTune`] object representing the newly created job.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]: if the request fails at the network layer.
/// - [`OpenAIError::DeserializeError`]: if the response fails to parse.
/// - [`OpenAIError::APIError`]: if OpenAI returns an error (e.g., invalid training file).
pub async fn create_fine_tune(
    client: &OpenAIClient,
    request: &CreateFineTuneRequest,
) -> Result<FineTune, OpenAIError> {
    let endpoint = "fine-tunes";
    post_json(client, endpoint, request).await
}

/// Lists all fine-tune jobs associated with the user's API key.
///
/// # Returns
///
/// A [`FineTuneList`] object containing all fine-tune jobs.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]
/// - [`OpenAIError::DeserializeError`]
/// - [`OpenAIError::APIError`]
pub async fn list_fine_tunes(client: &OpenAIClient) -> Result<FineTuneList, OpenAIError> {
    let endpoint = "fine-tunes";
    get_json(client, endpoint).await
}

/// Retrieves a fine-tune job by its ID (e.g. "ft-XXXXXXXX").
///
/// # Parameters
///
/// * `fine_tune_id` - The ID of the fine-tune job.
///
/// # Returns
///
/// A [`FineTune`] object with detailed information about the job.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]
/// - [`OpenAIError::DeserializeError`]
/// - [`OpenAIError::APIError`]
pub async fn retrieve_fine_tune(
    client: &OpenAIClient,
    fine_tune_id: &str,
) -> Result<FineTune, OpenAIError> {
    let endpoint = format!("fine-tunes/{}", fine_tune_id);
    get_json(client, &endpoint).await
}

/// Cancels a fine-tune job by its ID.
///
/// # Parameters
///
/// * `fine_tune_id` - The ID of the fine-tune job to cancel.
///
/// # Returns
///
/// The updated [`FineTune`] object with a status of "cancelled".
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]
/// - [`OpenAIError::DeserializeError`]
/// - [`OpenAIError::APIError`]
pub async fn cancel_fine_tune(
    client: &OpenAIClient,
    fine_tune_id: &str,
) -> Result<FineTune, OpenAIError> {
    let endpoint = format!("fine-tunes/{}/cancel", fine_tune_id);
    post_json::<(), FineTune>(client, &endpoint, &()).await
}

/// Lists events for a given fine-tune job (useful for seeing training progress).
///
/// # Parameters
///
/// * `fine_tune_id` - The ID of the fine-tune job.
///
/// # Returns
///
/// A list of [`FineTuneEvent`] objects, wrapped in a JSON list object.
///
/// # Errors
///
/// - [`OpenAIError::HTTPError`]
/// - [`OpenAIError::DeserializeError`]
/// - [`OpenAIError::APIError`]
pub async fn list_fine_tune_events(
    client: &OpenAIClient,
    fine_tune_id: &str,
) -> Result<FineTuneEventsList, OpenAIError> {
    let endpoint = format!("fine-tunes/{}/events", fine_tune_id);
    get_json(client, &endpoint).await
}

/// A helper struct for deserializing the result of `GET /v1/fine-tunes/{fine_tune_id}/events`.
#[derive(Debug, Deserialize)]
pub struct FineTuneEventsList {
    /// The object type, typically "list".
    pub object: String,
    /// The array of events.
    pub data: Vec<FineTuneEvent>,
}

/// Deletes a fine-tuned model (i.e., the actual model generated after successful fine-tuning).
///
/// **Note**: You can only delete models that you own or have permission to delete.
/// The fine-tuning job itself remains in the system for historical reference, but the model
/// can no longer be used once deleted.
///
/// # Parameters
///
/// * `model` - The name of the fine-tuned model to delete (e.g. "curie:ft-yourorg-2023-01-01-xxxx").
pub async fn delete_fine_tune_model(
    client: &OpenAIClient,
    model: &str,
) -> Result<DeleteFineTuneModelResponse, OpenAIError> {
    // Build the DELETE request
    let endpoint = format!("models/{}", model);
    let url = format!("{}/{}", client.base_url().trim_end_matches('/'), endpoint);

    let response = client
        .http_client
        .delete(&url)
        .bearer_auth(client.api_key())
        .send()
        .await?; // Network/HTTP-layer error if this fails

    // Check if the status code indicates success
    if !response.status().is_success() {
        // Attempt to parse a JSON error body in OpenAI’s format
        return Err(parse_error_response(response).await?);
    }

    // Otherwise, parse success body
    let response_body = response.json::<DeleteFineTuneModelResponse>().await?;
    Ok(response_body)
}
/// Response returned after deleting a fine-tuned model.
#[derive(Debug, Deserialize)]
pub struct DeleteFineTuneModelResponse {
    /// The object type, e.g., "model".
    pub object: String,
    /// The name of the deleted model.
    pub id: String,
    /// A message indicating the model was deleted.
    pub deleted: bool,
}

#[cfg(test)]
mod tests {
    /// # Tests for the `fine_tunes` module
    ///
    /// We use [`wiremock`](https://crates.io/crates/wiremock) to simulate OpenAI's Fine-tunes API,
    /// covering:
    /// 1. **create_fine_tune** – success & error
    /// 2. **list_fine_tunes** – success & error
    /// 3. **retrieve_fine_tune** – success & error
    /// 4. **cancel_fine_tune** – success & error
    /// 5. **list_fine_tune_events** – success & error
    /// 6. **delete_fine_tune_model** – success & error
    ///
    use super::*;
    use crate::config::OpenAIClient;
    use crate::error::OpenAIError;
    use serde_json::json;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_create_fine_tune_success() {
        let mock_server = MockServer::start().await;

        // Mock success response
        let success_body = json!({
            "id": "ft-abcdefgh",
            "object": "fine-tune",
            "created_at": 1673645000,
            "updated_at": 1673645200,
            "model": "curie",
            "fine_tuned_model": null,
            "status": "pending",
            "events": []
        });

        Mock::given(method("POST"))
            .and(path("/fine-tunes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateFineTuneRequest {
            training_file: "file-abc123".into(),
            model: Some("curie".into()),
            ..Default::default()
        };

        let result = create_fine_tune(&client, &req).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let fine_tune = result.unwrap();
        assert_eq!(fine_tune.id, "ft-abcdefgh");
        assert_eq!(fine_tune.status, "pending");
        assert_eq!(fine_tune.model, "curie");
        assert!(fine_tune.fine_tuned_model.is_none());
        assert_eq!(fine_tune.events.len(), 0);
    }

    #[tokio::test]
    async fn test_create_fine_tune_api_error() {
        let mock_server = MockServer::start().await;

        // Mock error
        let error_body = json!({
            "error": {
                "message": "Invalid training file",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("POST"))
            .and(path("/fine-tunes"))
            .respond_with(ResponseTemplate::new(400).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let req = CreateFineTuneRequest {
            training_file: "file-nonexistent".into(),
            ..Default::default()
        };

        let result = create_fine_tune(&client, &req).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Invalid training file"));
            }
            other => panic!("Expected APIError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_fine_tunes_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "object": "list",
            "data": [
                {
                    "id": "ft-abc123",
                    "object": "fine-tune",
                    "created_at": 1673645000,
                    "updated_at": 1673645200,
                    "model": "curie",
                    "fine_tuned_model": "curie:ft-yourorg-2023-01-01-xxxx",
                    "status": "succeeded",
                    "events": []
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path("/fine-tunes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_fine_tunes(&client).await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let list = result.unwrap();
        assert_eq!(list.object, "list");
        assert_eq!(list.data.len(), 1);
        let first = &list.data[0];
        assert_eq!(first.id, "ft-abc123");
        assert_eq!(first.status, "succeeded");
    }

    #[tokio::test]
    async fn test_list_fine_tunes_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Could not list fine-tunes",
                "type": "internal_server_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path("/fine-tunes"))
            .respond_with(ResponseTemplate::new(500).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_fine_tunes(&client).await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Could not list fine-tunes"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retrieve_fine_tune_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "id": "ft-xyz789",
            "object": "fine-tune",
            "created_at": 1673646000,
            "updated_at": 1673646200,
            "model": "curie",
            "fine_tuned_model": null,
            "status": "running",
            "events": []
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/fine-tunes/ft-xyz789$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_fine_tune(&client, "ft-xyz789").await;
        assert!(result.is_ok(), "Expected Ok, got: {:?}", result);

        let ft = result.unwrap();
        assert_eq!(ft.id, "ft-xyz789");
        assert_eq!(ft.status, "running");
    }

    #[tokio::test]
    async fn test_retrieve_fine_tune_api_error() {
        let mock_server = MockServer::start().await;
        let error_body = json!({
            "error": {
                "message": "Fine-tune not found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/fine-tunes/ft-000$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = retrieve_fine_tune(&client, "ft-000").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Fine-tune not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_cancel_fine_tune_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "id": "ft-abc123",
            "object": "fine-tune",
            "created_at": 1673647000,
            "updated_at": 1673647200,
            "model": "curie",
            "fine_tuned_model": null,
            "status": "cancelled",
            "events": []
        });

        Mock::given(method("POST"))
            .and(path_regex(r"^/fine-tunes/ft-abc123/cancel$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = cancel_fine_tune(&client, "ft-abc123").await;
        assert!(result.is_ok(), "Expected Ok, got {:?}", result);

        let ft = result.unwrap();
        assert_eq!(ft.id, "ft-abc123");
        assert_eq!(ft.status, "cancelled");
    }

    #[tokio::test]
    async fn test_cancel_fine_tune_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Cannot cancel a completed fine-tune",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("POST"))
            .and(path_regex(r"^/fine-tunes/ft-zzz/cancel$"))
            .respond_with(ResponseTemplate::new(400).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = cancel_fine_tune(&client, "ft-zzz").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Cannot cancel a completed fine-tune"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_list_fine_tune_events_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "object": "list",
            "data": [
                {
                    "object": "fine-tune-event",
                    "created_at": 1673648000,
                    "level": "info",
                    "message": "Job enqueued"
                },
                {
                    "object": "fine-tune-event",
                    "created_at": 1673648100,
                    "level": "info",
                    "message": "Job started"
                }
            ]
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/fine-tunes/ft-abc/events$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_fine_tune_events(&client, "ft-abc").await;
        assert!(result.is_ok(), "Expected Ok, got {:?}", result);

        let events_list = result.unwrap();
        assert_eq!(events_list.object, "list");
        assert_eq!(events_list.data.len(), 2);
        assert_eq!(events_list.data[0].message, "Job enqueued");
    }

    #[tokio::test]
    async fn test_list_fine_tune_events_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "No events found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("GET"))
            .and(path_regex(r"^/fine-tunes/ft-xyz/events$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = list_fine_tune_events(&client, "ft-xyz").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("No events found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_delete_fine_tune_model_success() {
        let mock_server = MockServer::start().await;

        let success_body = json!({
            "object": "model",
            "id": "curie:ft-yourorg-2023-01-01-xxxx",
            "deleted": true
        });

        Mock::given(method("DELETE"))
            .and(path_regex(r"^/models/curie:ft-yourorg-2023-01-01-xxxx$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = delete_fine_tune_model(&client, "curie:ft-yourorg-2023-01-01-xxxx").await;
        assert!(result.is_ok(), "Expected Ok, got {:?}", result);

        let del_resp = result.unwrap();
        assert_eq!(del_resp.object, "model");
        assert_eq!(del_resp.id, "curie:ft-yourorg-2023-01-01-xxxx");
        assert!(del_resp.deleted);
    }

    #[tokio::test]
    async fn test_delete_fine_tune_model_api_error() {
        let mock_server = MockServer::start().await;

        let error_body = json!({
            "error": {
                "message": "Model not found",
                "type": "invalid_request_error",
                "code": null
            }
        });

        Mock::given(method("DELETE"))
            .and(path_regex(r"^/models/doesnotexist$"))
            .respond_with(ResponseTemplate::new(404).set_body_json(error_body))
            .mount(&mock_server)
            .await;

        let client = OpenAIClient::builder()
            .with_api_key("test-key")
            .with_base_url(&mock_server.uri())
            .build()
            .unwrap();

        let result = delete_fine_tune_model(&client, "doesnotexist").await;
        match result {
            Err(OpenAIError::APIError { message, .. }) => {
                assert!(message.contains("Model not found"));
            }
            other => panic!("Expected APIError, got {:?}", other),
        }
    }
}
