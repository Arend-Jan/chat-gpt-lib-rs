# ChatGPT Rust Library
A Rust library for interacting with OpenAI's ChatGPT API. This library simplifies the process of making requests to the ChatGPT API and parsing responses.

## Features
* Easy to use interface for interacting with the ChatGPT API
* Strongly typed structures for request parameters and response data
* Support for serialization and deserialization using Serde

## Installation
Add the following line to your 'Cargo.toml' file under the '[dependencies]' section:
```toml
chat-gpt-lib-rs = "0.1.0"
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

## Documentation
For more details about the request parameters and response structure, refer to the [OpenAI API documentation](https://beta.openai.com/docs/api-reference/chat/create).

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
