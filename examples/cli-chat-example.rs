use chat_gpt_lib_rs::client::chat_input::Message;
use chat_gpt_lib_rs::client::error::ChatGPTError;
use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Model, Role};
use console::{style, StyledObject};
use dotenvy::dotenv;
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::io::{stdin, stdout, Write};
use std::iter::Skip;
use std::time::Duration;

/// This program is a command-line interface (CLI) example that demonstrates how to use the
/// `chat-gpt-lib-rs` library to interact with an AI model based on the GPT-3.5-Turbo architecture.
/// The CLI allows users to communicate with the AI model as if they are chatting with the Star Trek
/// computer.
///
/// The main function begins by loading environment variables from the `.env` file, including the
/// API key and icon usage setting. It then initializes the `ChatGPTClient` and message history with
/// a system message that sets the context for the AI model.
///
/// The program checks if there are any command-line arguments provided, and if so, it processes
/// the user input from these arguments as the first chat message. Afterwards, it enters the main
/// loop where user input is accepted and responses are generated.
///
/// A helper function, `process_user_input`, handles processing the user input, making the API call,
/// and displaying the computer's response. The function takes the user input, adds it to the message
/// history, prepares the `ChatInput` object, and then sends the API request. While waiting for the
/// API response, a spinner is displayed to indicate progress. Once the response is received, the
/// computer's response is extracted, displayed to the user, and added to the message history.
///
/// For an enhanced experience with icons, the terminal must use Nerd Fonts. This requirement
/// enables the program to display icons alongside text. https://www.nerdfonts.com/

// The main function, which is asynchronous due to the API call
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the logger
    env_logger::init();

    // Load the environment variables from the .env file
    dotenv().ok();

    // Get the API key and icon usage setting from the environment variables
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not found in .env");

    // Add USE_ICONS=true to your .env file, if your terminal is running with a
    // Nerd Font, so you get some pretty icons
    let use_icons = env::var("USE_ICONS")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        .eq("true");

    // Add VERBOSE_RETURN=true to your .env file, if you want te see the meta data
    // of the chat as well
    let verbose_response = env::var("VERBOSE_RESPONSE")
        .unwrap_or_else(|_| "false".to_string())
        .to_lowercase()
        .eq("true");

    // Initialize the ChatGPT client
    let client = ChatGPTClient::new(&api_key, "https://api.openai.com");

    // Initialize the message history with a system message
    let mut messages = vec![Message {
        role: Role::System,
        content: "You are a helpful assistant who provides solutions and examples in code blocks."
            .into(),
    }];

    // Check if any command line arguments are provided
    let mut args: Skip<env::Args> = env::args().skip(1);
    if let Some(first_arg) = args.next() {
        let user_message_content = args.fold(first_arg, |acc, arg| acc + " " + &arg);

        // Process the user input from command line arguments
        process_user_input(
            &client,
            &mut messages,
            user_message_content,
            verbose_response,
            use_icons,
        )
        .await?;
    }

    // Enter the main loop, where user input is accepted and responses are generated
    loop {
        // Display the input prompt with an optional icon
        let input_prompt: StyledObject<&str> = if use_icons {
            style("\u{f0ede}  Input: ").green()
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
            user_message_content,
            verbose_response,
            use_icons,
        )
        .await?;
    }
}

///
/// # Arguments
///
/// * `client` - A reference to the ChatGPT client.
/// * `messages` - A mutable reference to the message history.
/// * `user_message_content` - The user's message content.
/// * `verbose_response` - A boolean indicating whether to display metadata.
/// * `use_icons` - A boolean indicating whether to use icons in the display.
async fn process_user_input(
    client: &ChatGPTClient,
    messages: &mut Vec<Message>,
    user_message_content: String,
    verbose_response: bool,
    use_icons: bool,
) -> Result<(), ChatGPTError> {
    // Add the user message to the message history
    messages.push(Message {
        role: Role::User,
        content: user_message_content.trim().into(),
    });

    // Prepare the ChatInput object for the API call
    let input = ChatInput {
        model: Model::Gpt_4o,
        messages: messages.clone(),
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
        let result = client.chat(input).await;
        spinner.finish_and_clear();
        result?
    };

    // Extract the assistant's message from the API response
    let assistant_message = &chat.choices[0].message.content;

    // Determine labels based on the use_icons flag
    let computer_label = if use_icons {
        style("\u{f12ca}  Computer: ").blue()
    } else {
        style("Computer: ").blue()
    };

    let metadata_label = if use_icons {
        "\u{f121}  Metadata:"
    } else {
        "Metadata:"
    };

    // If verbose response is enabled, display additional metadata
    if verbose_response {
        println!(
            "{}",
            style(format!(
                "{}\n   ID: {}\n   Created: {}\n   Model: {}\n   Usage: {:?}",
                metadata_label, chat.id, chat.created, chat.model, chat.usage
            ))
            .yellow()
        );
    }

    // Display the computer's response
    let computer_response = style(assistant_message.clone());
    println!("{}{}", computer_label, computer_response);

    // Add the assistant's message to the message history
    messages.push(Message {
        role: Role::Assistant,
        content: assistant_message.clone(),
    });

    Ok(())
}
