//! An example showcasing how to use the embeddings api
//!
//! To run this example:
//! ```bash
//! cargo run --example embeddings
//! ```

use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::api_resources::embeddings::{
    CreateEmbeddingsRequest, EmbeddingsInput, create_embeddings,
};
use chat_gpt_lib_rs::api_resources::models::Model;
use chat_gpt_lib_rs::error::OpenAIError;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY

    let request = CreateEmbeddingsRequest {
        model: Model::TextEmbeddingAda002,
        input: EmbeddingsInput::String("Hello world".to_string()),
        user: None,
    };

    let response = create_embeddings(&client, &request).await?;
    for (i, emb) in response.data.iter().enumerate() {
        println!("Embedding #{}: vector size = {}", i, emb.embedding.len());
    }
    println!("Model used: {:?}", response.model);
    if let Some(usage) = &response.usage {
        println!(
            "Usage => prompt_tokens: {}, total_tokens: {}",
            usage.prompt_tokens, usage.total_tokens
        );
    }

    Ok(())
}
