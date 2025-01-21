//! An example showcasing how to use the embeddings api
//!
//! To run this example:
//! ```bash
//! cargo run --example embeddings
//! ```

use chat_gpt_lib_rs::api_resources::embeddings::{
    create_embeddings, CreateEmbeddingsRequest, EmbeddingsInput,
};
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    let client = OpenAIClient::new(None)?; // Reads API key from OPENAI_API_KEY

    let request = CreateEmbeddingsRequest {
        model: "text-embedding-ada-002".to_string(),
        input: EmbeddingsInput::String("Hello world".to_string()),
        user: None,
    };

    let response = create_embeddings(&client, &request).await?;
    for (i, emb) in response.data.iter().enumerate() {
        println!("Embedding #{}: vector size = {}", i, emb.embedding.len());
    }
    println!("Model used: {}", response.model);
    if let Some(usage) = &response.usage {
        println!(
            "Usage => prompt_tokens: {}, total_tokens: {}",
            usage.prompt_tokens, usage.total_tokens
        );
    }

    Ok(())
}
