// Import the necessary crates and modules
use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model, Role};
use dotenvy::dotenv;
use std::io::{stdin, stdout, Write};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the environment variables from the .env file
    dotenv().ok();

    // Retrieve the OpenAI API key from the environment variables
    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found in environment");

    // Create a new instance of the ChatGPTClient
    let client = ChatGPTClient::new(&api_key, "https://api.openai.com");

    // Create a vector of messages with an initial system message
    let mut messages = vec![Message {
        role: Role::System,
        content: "You are an AI that can answer any question.".to_string(),
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
            content: user_input.trim().to_string(),
        });

        // Define the input for the ChatGPTClient
        let input = ChatInput {
            model: Model::Gpt3_5Turbo,  // Set the GPT-3.5 Turbo model
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
