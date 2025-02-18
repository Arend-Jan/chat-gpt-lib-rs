//! An example showcasing how to create chat-based completions using the streaming functionality
//! of the OpenAI Chat Completions API.
//!
//! To run this example:
//! ```bash
//! cargo run --example chat_streaming
//! ```

use futures_util::StreamExt; // Brings the `next()` method into scope.
use std::io::{self, Write};

use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion_stream, ChatMessage, ChatRole, CreateChatCompletionRequest,
};
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Optionally load environment variables from a .env file.
    dotenvy::dotenv().ok();

    // Create a new client; the API key is taken from the OPENAI_API_KEY environment variable.
    let client = OpenAIClient::new(None)?;

    // Build a chat request. Note that we enable streaming by setting `stream: Some(true)`.
    let request = CreateChatCompletionRequest {
        model: "gpt-3.5-turbo".into(),
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
        stream: Some(true), // Enable streaming responses.
        ..Default::default()
    };

    println!("Sending chat completion streaming request...");

    // Request a stream of partial chat responses.
    let mut stream = create_chat_completion_stream(&client, &request).await?;

    println!("\n\n\n*** Streaming completion: *** \n");
    let mut full_response = String::new();

    // Iterate over each streaming chunk.
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // For streaming chat responses, the partial update is contained in the delta field.
                // Check the first choice for a delta update.
                if let Some(choice) = chunk.choices.first() {
                    if let Some(content) = &choice.delta.content {
                        // Print the partial text without adding a newline.
                        print!("{}", content);
                        // Flush stdout immediately so the text appears incrementally.
                        io::stdout().flush().unwrap();
                        full_response.push_str(content);
                    }
                }
            }
            Err(e) => {
                eprintln!("Stream error: {:?}", e);
            }
        }
    }

    println!("\n\nFull chat response:\n{}", full_response);

    Ok(())
}
