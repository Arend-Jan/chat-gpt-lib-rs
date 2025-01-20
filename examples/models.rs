//! An example showcasing how to list and retrieve OpenAI models.
//!
//! To run this example:
//! ```bash
//! cargo run --example models
//! ```

use chat_gpt_lib_rs::api_resources::models;
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    // This is helpful if you don't want to expose the API key directly in your code.
    dotenvy::dotenv().ok();

    // Create a new client; this will look for the OPENAI_API_KEY environment variable.
    let client = OpenAIClient::new(None)?;

    // List all available models
    println!("Listing available models...");
    let all_models = models::list_models(&client).await?;
    println!("Found {} models:", all_models.len());
    for model in &all_models {
        println!("  - {} (owned by {})", model.id, model.owned_by);
    }

    // If we have at least one model, retrieve details about the first one
    if let Some(first_model) = all_models.first() {
        println!("\nRetrieving details about model '{}'", first_model.id);
        let detailed_model = models::retrieve_model(&client, &first_model.id).await?;
        println!("Model details: {:?}", detailed_model);
    } else {
        println!("No models found.");
    }

    Ok(())
}
