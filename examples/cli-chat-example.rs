//! An example showcasing how to create chat-based completions using the OpenAI Chat Completions API.
//!
//! To run this example:
//! ```bash
//! cargo run --example chat
//! ```

use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion, ChatMessage, ChatRole, CreateChatCompletionRequest,
};
use chat_gpt_lib_rs::error::OpenAIError;
use chat_gpt_lib_rs::OpenAIClient;
use console::{style, StyledObject};
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::io::{stdin, stdout, Write};
use std::iter::Skip;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Load environment variables from a .env file, if present (optional).
    dotenvy::dotenv().ok();

    // Create a new client; this will look for the OPENAI_API_KEY environment variable.
    // Alternatively, you can provide an explicit API key via `OpenAIClient::new(Some("sk-XXXX"))`.
    let client = OpenAIClient::new(None)?;

    // Add USE_ICONS=true to your .env file, if your terminal is running with a
    // Nerd Font, so you get some pretty icons
    let use_icons = env::var("USE_ICONS")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        .eq("true");

    let model = env::var("CHAT_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

    let system_prompt = env::var("SYSTEM_PROMPT").unwrap_or_else(|_| {
        "You are a high quality tech lead and are specialized in idiomatic Rust".to_string()
    });

    let max_tokens: Option<u32> = env::var("MAX_TOKENS")
        .ok()
        .and_then(|val| val.parse::<u32>().ok())
        .or(Some(150));

    let temperature: Option<f64> = env::var("TEMPERATURE")
        .ok()
        .and_then(|val| val.parse::<f64>().ok())
        .or(Some(0.7));

    // Initialize the message history with a system message
    let mut messages = vec![ChatMessage {
        role: ChatRole::System,
        content: system_prompt,
        name: None,
    }];

    // Check if any command line arguments are provided
    let mut args: Skip<env::Args> = env::args().skip(1);
    if let Some(first_arg) = args.next() {
        let user_message_content = args.fold(first_arg, |acc, arg| acc + " " + &arg);

        // Process the user input from command line arguments
        process_user_input(
            &client,
            &mut messages,
            &user_message_content,
            &model,
            max_tokens,
            temperature,
        )
        .await?;
    }

    // Enter the main loop, where user input is accepted and responses are generated
    loop {
        // Display the input prompt with an optional icon
        let input_prompt: StyledObject<&str> = if use_icons {
            style("\u{f0ede} Input: ").green()
        } else {
            style("Input: ").green()
        };
        print!("{}", input_prompt);
        stdout().flush().unwrap();

        // Read the user input
        let mut user_message_content = String::new();
        stdin().read_line(&mut user_message_content).unwrap();

        // Process the user input and generate a response
        process_user_input(
            &client,
            &mut messages,
            &user_message_content,
            &model,
            max_tokens,
            temperature,
        )
        .await?;
    }
}

async fn process_user_input(
    client: &OpenAIClient,
    messages: &mut Vec<ChatMessage>,
    user_message_content: &String,
    model: &String,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
) -> Result<(), OpenAIError> {
    // Add the user message to the message history
    messages.push(ChatMessage {
        role: ChatRole::User,
        content: user_message_content.trim().to_string(),
        name: None,
    });

    // Prepare the ChatInput object for the API call
    let request = CreateChatCompletionRequest {
        model: model.clone(),
        messages: messages.clone(),
        max_tokens,
        temperature,
        ..Default::default()
    };

    // Set up a spinner to display while waiting for the API response
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{spinner:.green} Processing...")
            .unwrap(),
    );

    // Make the API call and store the result
    let chat = {
        spinner.enable_steady_tick(Duration::from_millis(100));
        //let result = client.create(input).await;
        let result = create_chat_completion(&client, &request).await?;
        spinner.finish_and_clear();
        result
    };

    // Extract the assistant's message from the API response
    let assistant_message = &chat.choices[0].message.content;

    // Display the computer's response with an optional icon
    let computer_label: StyledObject<&str> = if env::var("USE_ICONS")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        .eq("true")
    {
        style("\u{f12ca} Computer: ").color256(39)
    } else {
        style("Computer: ").color256(39)
    };
    let computer_response: StyledObject<String> = style(assistant_message.clone());

    println!("{}{}", computer_label, computer_response);

    // Add the assistant's message to the message history
    messages.push(ChatMessage {
        role: ChatRole::Assistant,
        content: assistant_message.clone(),
        name: None,
    });

    Ok(())
}
