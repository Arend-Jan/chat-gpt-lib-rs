//! An example showcasing how to create text completions using the OpenAI Completions API.
//!
//! To run this example:
//! ```bash
//! cargo run --example completions
//! ```

use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::api_resources::completions::{
    CreateCompletionRequest, PromptInput, create_completion_stream,
};
use chat_gpt_lib_rs::error::OpenAIError;
use futures_util::StreamExt;
use std::io::{self, Write}; // Import flush functionality // Brings `next()` into scope.

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    // If you don't have OPENAI_API_KEY set in your environment, you can manually assign:
    // let api_key = "sk-XXXX".to_string();
    // let client = OpenAIClient::new(Some(api_key))?;

    // Otherwise, use None to read from the OPENAI_API_KEY env variable:
    let client = OpenAIClient::new(None)?;

    // We’ll create a simple prompt to ask for a short advertisement
    // about ice cream, as an example:
    let prompt_text = "Write a advertisement for a new flavor of ice cream.";

    // Build a request using a completions model (e.g. text-davinci-003)
    let request = CreateCompletionRequest {
        model: "gpt-3.5-turbo-instruct".into(),
        prompt: Some(PromptInput::String(prompt_text.to_owned())),
        max_tokens: Some(512),
        stream: Some(true), // Enable streaming responses.
        ..Default::default()
    };

    println!("Sending a completion request for prompt: {:?}", prompt_text);

    // Request a stream of partial responses.
    let mut stream = create_completion_stream(&client, &request).await?;

    println!("\n\n\n*** Streaming completion: *** \n");

    // Accumulate the generated text.
    let mut full_text = String::new();

    // Iterate over each streaming chunk.
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Extract text from the first choice, if available.
                if let Some(choice) = chunk.choices.first() {
                    // Print the partial text (without the extra struct formatting).
                    print!("{}", choice.text);
                    // Flush stdout immediately so that the output is visible.
                    io::stdout().flush().unwrap();
                    full_text.push_str(&choice.text);
                }
            }
            Err(e) => {
                eprintln!("Stream error: {:?}", e);
            }
        }
    }

    println!("\n\n\n*** Full completion: *** \n{}", full_text);

    Ok(())
}
