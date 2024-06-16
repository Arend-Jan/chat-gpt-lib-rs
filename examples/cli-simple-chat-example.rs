use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model, Role};
use dotenvy::dotenv;
use std::env;
use std::error::Error;
use std::io::{stdin, stdout, Write};

/// Main entry point for the Chat GPT client application.
///
/// This application interacts with the OpenAI GPT model to answer user questions in a chat interface.
/// It demonstrates the basic setup for asynchronous communication with the OpenAI API, including loading
/// API keys from environment variables, creating a chat client, and processing user input.
///
/// Environment Variables:
/// - `OPENAI_API_KEY`: The API key for accessing OpenAI's services. Must be set in a `.env` file or the environment.
///
/// Dependencies:
/// - `chat_gpt_lib_rs`: A library for interfacing with the Chat GPT model.
/// - `dotenvy`: For loading environment variables from a `.env` file.
///
/// Error Handling:
/// This application uses simple error handling with `expect` for demonstration. In a production environment,
/// more robust error handling should be implemented.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load the environment variables from the .env file
    dotenv().ok();

    // Retrieve the OpenAI API key from the environment variables
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found in environment");

    // Create a new instance of the ChatGPTClient
    let client = ChatGPTClient::new(&api_key, "https://api.openai.com");

    // Create a vector of messages with an initial system message
    let mut messages = vec![Message {
        role: Role::System,
        content: "You are an AI that can answer any question.".into(),
    }];

    // Start an input loop
    loop {
        // Prompt the user for input
        print!("Enter your question: ");
        stdout().flush().unwrap();

        // Read the user's input
        let mut user_input = String::new();
        stdin().read_line(&mut user_input).unwrap();

        // Add the user's message to the messages vector
        messages.push(Message {
            role: Role::User,
            content: user_input.trim().into(),
        });

        // Define the input for the ChatGPTClient
        let input = ChatInput {
            model: Model::Gpt_4o,       // Consider making this configurable
            messages: messages.clone(), // Pass in the messages vector
            ..Default::default()
        };

        // Call the chat method on the ChatGPTClient with the input
        let response = client.chat(input).await?;

        // Retrieve the AI's response from the first choice
        let ai_message = &response.choices[0].message.content;

        // Print the AI's response to the console
        println!("AI Response: {}", ai_message);

        // Add the AI's message to the messages vector
        messages.push(Message {
            role: Role::Assistant,
            content: ai_message.clone(),
        });
    }
}
