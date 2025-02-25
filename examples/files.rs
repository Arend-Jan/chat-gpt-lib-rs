//! An example demonstrating how to upload, list, retrieve, download, and delete files
//! using the OpenAI Files API.
//!
//! To run this example:
//! ```bash
//! cargo run --example files
//! ```
//!
//! **Important**: This assumes you have a file named `file-abc123.jsonl` in the `examples/` folder,
//! containing data in the JSONL format. You can rename or edit this file as needed.

use std::path::PathBuf;

use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::api_resources::files::{
    UploadFilePurpose, delete_file, list_files, retrieve_file_content, retrieve_file_metadata,
    upload_file,
};
use chat_gpt_lib_rs::error::OpenAIError;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    // Create a new client; reads OPENAI_API_KEY from the environment by default.
    let client = OpenAIClient::new(None)?;

    // Path to our example JSONL file in the "examples" folder.
    // Adjust if your file is located elsewhere or has a different name.
    let file_path = PathBuf::from("examples").join("file-abc123.jsonl");

    println!("Uploading file '{}'", file_path.display());
    // Upload the file with purpose "fine-tune"
    let uploaded_file = upload_file(&client, &file_path, UploadFilePurpose::FineTune).await?;
    println!("Uploaded File ID: {}", uploaded_file.id);
    println!("File purpose: {}", uploaded_file.purpose);

    // List all files and show the newly uploaded file among them
    println!("\nListing all files:");
    let files_list = list_files(&client).await?;
    for f in files_list.data {
        println!(" - File: {} (ID: {})", f.filename, f.id);
    }

    // Retrieve metadata about the uploaded file
    println!("\nRetrieving metadata for the uploaded file...");
    let file_meta = retrieve_file_metadata(&client, &uploaded_file.id).await?;
    println!(
        "File '{}' has status: {:?}",
        file_meta.filename, file_meta.status
    );

    // Optionally download the file content to verify its contents
    println!("\nDownloading the file content...");
    let content_bytes = retrieve_file_content(&client, &uploaded_file.id).await?;
    println!("File size (bytes): {}", content_bytes.len());
    println!(
        "File content (parsed in UTF-8): \n{}",
        String::from_utf8(content_bytes).unwrap_or("Non UTF-8 bytes".to_string())
    );

    // Delete the file if you no longer need it
    println!("\nDeleting the file...");
    let delete_response = delete_file(&client, &uploaded_file.id).await?;
    println!(
        "File '{}' deleted: {}",
        delete_response.id, delete_response.deleted
    );

    Ok(())
}
