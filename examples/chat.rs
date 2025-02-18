//! An example showcasing how to create chat-based completions using the OpenAI Chat Completions API.
//!
//! To run this example:
//! ```bash
//! cargo run --example chat
//! ```

use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion, ChatMessage, ChatRole, CreateChatCompletionRequest,
};
use chat_gpt_lib_rs::api_resources::models::Model;
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    // Create a new client; this will look for the OPENAI_API_KEY environment variable.
    // Alternatively, you can provide an explicit API key via `OpenAIClient::new(Some("sk-XXXX"))`.
    let client = OpenAIClient::new(None)?;

    // Build a chat request, including a system message and a user prompt
    let request = CreateChatCompletionRequest {
        model: Model::Gpt4,
        messages: vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a cheerful and friendly assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Could you write me a quick recipe for chocolate chip cookies?"
                    .to_string(),
                name: None,
            },
        ],
        max_tokens: Some(150),
        temperature: Some(0.7),
        ..Default::default()
    };

    println!("Sending chat completion request...");

    // Call our library function to create a chat completion
    let response = create_chat_completion(&client, &request).await?;

    // Print out each choice. Typically `n = 1`, so there's one main choice.
    for (i, choice) in response.choices.iter().enumerate() {
        println!("\n== Chat Choice {} ==", i);
        println!("Assistant: {}", choice.message.content);
        if let Some(reason) = &choice.finish_reason {
            println!("Finish reason: {}", reason);
        }
    }

    // If usage info is available, print it
    if let Some(usage) = &response.usage {
        println!(
            "\nUsage => prompt_tokens: {}, completion_tokens: {}, total_tokens: {}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
