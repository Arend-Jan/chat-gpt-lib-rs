//! An example showcasing how to classify text against OpenAI's content moderation policies
//! using the Moderations API.
//!
//! To run this example:
//! ```bash
//! cargo run --example moderations
//! ```

use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::api_resources::moderations::{
    CreateModerationRequest, ModerationsInput, create_moderation,
};
use chat_gpt_lib_rs::error::OpenAIError;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    // Create a new client; this will look for the OPENAI_API_KEY environment variable.
    let client = OpenAIClient::new(None)?;

    let moderation_text = "I hate you and want to harm you.".to_string();

    println!("Text to moderate: {}", &moderation_text);

    // Create a Moderations request for a single piece of text.
    // We can also provide multiple texts with `ModerationsInput::Strings(...)`.
    let request = CreateModerationRequest {
        input: ModerationsInput::String(moderation_text),
        // Optionally, specify a model like "text-moderation-latest" or "text-moderation-stable":
        model: None,
    };

    println!("Sending a moderation request...");

    // Call the Moderations API
    let response = create_moderation(&client, &request).await?;

    // The response contains one result per input. Here we only have one input.
    for (i, result) in response.results.iter().enumerate() {
        println!("\n== Moderation Result {} ==", i);
        println!("Flagged: {}", result.flagged);

        println!("Categories:");
        println!("  hate: {}", result.categories.hate);
        println!("  hate/threatening: {}", result.categories.hate_threatening);
        println!("  self-harm: {}", result.categories.self_harm);
        println!("  sexual: {}", result.categories.sexual);
        println!("  sexual/minors: {}", result.categories.sexual_minors);
        println!("  violence: {}", result.categories.violence);
        println!("  violence/graphic: {}", result.categories.violence_graphic);

        println!("Scores:");
        println!("  hate: {}", result.category_scores.hate);
        println!(
            "  hate/threatening: {}",
            result.category_scores.hate_threatening
        );
        println!("  self-harm: {}", result.category_scores.self_harm);
        println!("  sexual: {}", result.category_scores.sexual);
        println!("  sexual/minors: {}", result.category_scores.sexual_minors);
        println!("  violence: {}", result.category_scores.violence);
        println!(
            "  violence/graphic: {}",
            result.category_scores.violence_graphic
        );
    }

    Ok(())
}
