//! An end-to-end example showcasing how to manage fine-tune jobs using the OpenAI API.
//!
//! **Important**: This example assumes you have already uploaded a training file using the
//! Files API and obtained its file ID (e.g. "file-abc123"). Without a valid file, your
//! request to create a fine-tune will fail.
//!
//! **Steps**:
//! 1. Create a fine-tune job with an existing training file ID.
//! 2. List all fine-tune jobs to confirm creation.
//! 3. Retrieve the newly created fine-tune job by its ID.
//! 4. (Optional) List the fine-tune’s events to view training progress.
//! 5. (Optional) Cancel a pending fine-tune if needed.
//! 6. (Optional) Once training is complete, delete the resulting fine-tuned model if you wish.
//!
//! To run this example:
//! ```bash
//! cargo run --example fine_tunes
//! ```

use chat_gpt_lib_rs::api_resources::fine_tunes::{
    cancel_fine_tune, create_fine_tune, delete_fine_tune_model, list_fine_tune_events,
    list_fine_tunes, retrieve_fine_tune, CreateFineTuneRequest, FineTune,
};
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;
use std::env;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file if present (optional).
    dotenvy::dotenv().ok();

    // Create a new OpenAI client; will look for OPENAI_API_KEY in the environment.
    let client = OpenAIClient::new(None)?;

    // -------------------------------------------------------------------------
    // 1. Create a fine-tune job
    // -------------------------------------------------------------------------
    //
    // You must have previously uploaded a JSONL training file using the Files API.
    // For example:
    //   chat_gpt_lib_rs::api_resources::files::upload_file(...)  (not shown here)
    //
    // Once you have an uploaded file ID (e.g., "file-abc123"), put it here:
    let training_file_id =
        env::var("TRAINING_FILE_ID").unwrap_or_else(|_| "file-abc123".to_string());

    let create_request = CreateFineTuneRequest {
        // The ID of the training file you previously uploaded
        training_file: training_file_id,
        // Optional: specify a base model like "curie", "davinci", or "ada".
        // If not provided, it defaults to "curie" as per the docs.
        model: Some("curie".to_string()),
        // You can customize other parameters too, like n_epochs, batch_size, suffix, etc.
        ..Default::default()
    };

    println!("Creating a new fine-tune job...");
    let fine_tune_response: FineTune = create_fine_tune(&client, &create_request).await?;
    println!("Fine-tune created: ID = {}", fine_tune_response.id);
    println!("Status = {}", fine_tune_response.status);

    // -------------------------------------------------------------------------
    // 2. List all fine-tune jobs
    // -------------------------------------------------------------------------
    let all_fine_tunes = list_fine_tunes(&client).await?;
    println!(
        "\nListing all fine-tunes ({})...",
        all_fine_tunes.data.len()
    );
    for (i, ft) in all_fine_tunes.data.iter().enumerate() {
        println!(
            "{}. ID={} | Status={} | Model={:?}",
            i + 1,
            ft.id,
            ft.status,
            ft.fine_tuned_model
        );
    }

    // -------------------------------------------------------------------------
    // 3. Retrieve the newly created fine-tune job by its ID
    // -------------------------------------------------------------------------
    println!("\nRetrieving the newly created fine-tune...");
    let retrieved_ft = retrieve_fine_tune(&client, &fine_tune_response.id).await?;
    println!(
        "Retrieved fine-tune: ID={} | Status={} | fine_tuned_model={:?}",
        retrieved_ft.id, retrieved_ft.status, retrieved_ft.fine_tuned_model
    );

    // -------------------------------------------------------------------------
    // 4. (Optional) List the events for this fine-tune job
    // -------------------------------------------------------------------------
    println!("\nListing events for this fine-tune job...");
    let events_list = list_fine_tune_events(&client, &fine_tune_response.id).await?;
    for (i, event) in events_list.data.iter().enumerate() {
        println!("Event #{}: [Level: {}] {}", i, event.level, event.message);
    }
    if events_list.data.is_empty() {
        println!("No events found yet (the job might still be queueing or in progress).");
    }

    // -------------------------------------------------------------------------
    // 5. (Optional) Cancel the fine-tune if it's still running
    // -------------------------------------------------------------------------
    // If the job is still in progress (e.g., "pending", "running"), and you decide not to
    // proceed, you can cancel it:
    //
    // println!("\nCancelling the fine-tune job...");
    // let cancelled_ft = cancel_fine_tune(&client, &fine_tune_response.id).await?;
    // println!(
    //     "Cancelled fine-tune: ID={} | Status={}",
    //     cancelled_ft.id, cancelled_ft.status
    // );
    //
    // Uncomment the lines above to try out the cancel functionality.

    // -------------------------------------------------------------------------
    // 6. (Optional) Delete the resulting fine-tuned model after it completes
    // -------------------------------------------------------------------------
    //
    // Once the status is "succeeded", you'll see a `fine_tuned_model` name, something like:
    // "curie:ft-yourorg-2023-09-15-xxxxxxx"
    //
    // If you want to remove that model from your account (and can do so, e.g. you own the model),
    // you can delete it:
    //
    // if let Some(ref model_name) = retrieved_ft.fine_tuned_model {
    //     // Make sure the job actually succeeded before trying to delete, otherwise you'll get an error.
    //     if retrieved_ft.status == "succeeded" {
    //         println!("\nDeleting the fine-tuned model: {}...", model_name);
    //         let delete_response = delete_fine_tune_model(&client, model_name).await?;
    //         println!(
    //             "Model deletion response => object={}, id={}, deleted={}",
    //             delete_response.object, delete_response.id, delete_response.deleted
    //         );
    //     } else {
    //         println!(
    //             "\nThe job has not succeeded yet (status={}). Cannot delete model: {}",
    //             retrieved_ft.status,
    //             model_name
    //         );
    //     }
    // }

    println!("\nExample completed. Check the logs above for details.");
    Ok(())
}
