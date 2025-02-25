//! An example showcasing how to create text completions using the OpenAI Completions API.
//!
//! To run this example:
//! ```bash
//! cargo run --example completions
//! ```

use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::api_resources::completions::{
    CreateCompletionRequest, PromptInput, create_completion,
};
use chat_gpt_lib_rs::error::OpenAIError;

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
    let prompt_text = "Write a short advertisement for a new flavor of ice cream.";

    // Build our request for text completions
    let request = CreateCompletionRequest {
        model: "gpt-3.5-turbo-instruct".into(),
        prompt: Some(PromptInput::String(prompt_text.to_owned())),
        max_tokens: Some(50),
        temperature: Some(0.7),
        // Additional fields are optional, so we can leave them at default:
        ..Default::default()
    };

    println!(
        "Sending a completion request for prompt: \"{}\"",
        prompt_text
    );

    // Call our library function to create a completion
    let response = create_completion(&client, &request).await?;

    // Print out each completion choice (though typically we just have one)
    for (i, choice) in response.choices.iter().enumerate() {
        println!("Choice {}:\n{}", i, choice.text);
    }

    // Optional: Print token usage info if available
    if let Some(usage) = response.usage {
        println!(
            "Usage: prompt_tokens={}, completion_tokens={}, total_tokens={}",
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
        );
    }

    Ok(())
}
