# ChatGPT Rust Library
A Rust library for interacting with OpenAI's ChatGPT API. This library simplifies the process of making requests to the ChatGPT API and parsing responses.

## Features
* Easy to use interface for interacting with the ChatGPT API
* Strongly typed structures for request parameters and response data
* Support for serialization and deserialization using Serde
* An example CLI chat application that demonstrates library usage

## Installation
Add the following line to your 'Cargo.toml' file under the '[dependencies]' section:
```toml
chat-gpt-lib-rs = "0.1.3"
```
Then, run cargo build to download and compile the dependencies.

## Usage
First, import the necessary components:
```rust
use chat_gpt_lib_rs::{ChatGPTClient, ChatInput, Message, Model, Role};
```
Next, create a new client with your API key:
```rust
let api_key = "your_api_key_here";
let base_url = "https://api.openai.com";
let client = ChatGPTClient::new(api_key, base_url);
```
To send a chat message, create a ChatInput structure and call the chat method:
```rust
let chat_input = ChatInput {
    model: Model::Gpt3_5Turbo,
    messages: vec![
        Message {
            role: Role::System,
            content: "You are a helpful assistant.".to_string(),
        },
        Message {
            role: Role::User,
            content: "Who won the world series in 2020?".to_string(),
        },
    ],
    ..Default::default()
};

let response = client.chat(chat_input).await.unwrap();
```
The response will be a 'ChatResponse' structure containing the API response data.

## Example CLI Chat Application
An example CLI chat application is provided in the examples folder. The example, named cli-chat-example.rs, demonstrates how to use the chat-gpt-lib-rs library to interact with an AI model based on the GPT-3 architecture through a command-line interface.

To run the example, first set your OPENAI_API_KEY in the .env file or as an environment variable, and then execute the following command:
```sh
cargo run --example cli-chat-example
```
Optionally, you can provide initial user input as a command-line argument:
```sh
cargo run --example cli-chat-example "Hello, computer!"
```
For an enhanced experience with icons, use a terminal that supports [Nerd Fonts](https://www.nerdfonts.com/).

## Documentation
For more details about the request parameters and response structure, refer to the [OpenAI API documentation](https://beta.openai.com/docs/api-reference/chat/create).

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
