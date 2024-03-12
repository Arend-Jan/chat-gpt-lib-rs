[![Crates.io](https://img.shields.io/crates/v/chat-gpt-lib-rs.svg)](https://crates.io/crates/chat-gpt-lib-rs)
[![Documentation](https://docs.rs/chat-gpt-lib-rs/badge.svg)](https://docs.rs/chat-gpt-lib-rs/)
[![Codecov](https://codecov.io/github/arend-jan/chat-gpt-lib-rs/coverage.svg?branch=main)](https://codecov.io/gh/arend-jan/chat-gpt-lib-rs)
[![Dependency status](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs/status.svg)](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs)

# ChatGPT Rust Library
A Rust library for interacting with OpenAI's ChatGPT API. This library simplifies the process of making requests to the ChatGPT API and parsing responses.

## Features
* Easy to use interface for interacting with the ChatGPT API
* Strongly typed structures for request parameters and response data
* Support for serialization and deserialization using Serde
* An example CLI chat application that demonstrates library usage
* An token estimation functionality

## Installation
Add the following line to your 'Cargo.toml' file under the '[dependencies]' section:
```toml
chat-gpt-lib-rs = "<put here the latest and greatest version number>"
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
    model: Model::Gpt_4Turbo,
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
Two example CLI chat applications are provided in the examples folder:

### Simple Chat Application
The cli-simple-chat-example.rs demonstrates how to use the chat-gpt-lib-rs library to interact with an AI model based on the GPT-3 architecture through a command-line interface. To run the example, first set your OPENAI_API_KEY in the .env file or as an environment variable, and then execute the following command:
```sh
cargo run --example cli-simple-chat-example
```
The example will prompt the user to enter a question, and the AI chatbot will respond with an answer. The conversation will continue until the user exits the program.

Optionally, you can provide initial user input as a command-line argument:

```sh
cargo run --example cli-simple-chat-example "Hello, computer!"
```
### Fancy Chat Application
The cli-chat-example.rs demonstrates how to use the chat-gpt-lib-rs library to create an interactive AI chatbot with a command-line interface. To run the example, first set your OPENAI_API_KEY in the .env file or as an environment variable, and then execute the following command:
```sh
cargo run --example cli-chat-example
```
The example will prompt the user to enter a message, and the AI chatbot will respond with an answer. The conversation will continue until the user exits the program.

Optionally, you can provide initial user input as a command-line argument:

```sh
cargo run --example cli-chat-example "Hello, computer!"
```
For an enhanced experience with icons, use a terminal that supports [Nerd Fonts](https://www.nerdfonts.com/). To enable this feature set you USE_ICONS=true in the .env file or as en environment variable.

## Documentation
For more details about the request parameters and response structure, refer to the [OpenAI API documentation](https://beta.openai.com/docs/api-reference/chat/create).


## Contributing

We welcome contributions to the `chat-gpt-lib-rs` project! Whether it's reporting bugs, proposing new features, improving documentation, or contributing code, your help is greatly appreciated. Here's how you can contribute:

1. **Fork the Repository**: Start by forking the `chat-gpt-lib-rs` repository to your own GitHub account. This will create a copy of the repository that you can modify without affecting the original project.
2. **Create a Branch**: In your forked repository, create a new branch for the changes you want to make. This helps keep your changes separate from other changes and makes it easier to merge your changes later.
3. **Make Your Changes**: Make your changes in the new branch. This could be fixing a bug, adding a new feature, improving documentation, or any other changes you think would improve the project.
4. **Test Your Changes**: Make sure your changes work as expected and don't introduce any new bugs. If the project has a test suite, make sure your changes pass all the tests.
5. **Submit a Pull Request**: Once you're happy with your changes, submit a pull request to merge your branch into the main `chat-gpt-lib-rs` repository. In your pull request, describe the changes you made and why you think they should be included in the project.
6. **Address Review Feedback**: After you submit a pull request, other contributors to the project may review your changes and provide feedback. Be prepared to make additional changes or answer questions about your changes.

Remember, contributions to open source projects like `chat-gpt-lib-rs` are a collaborative effort. Be respectful and patient with other contributors, and remember that everyone is working together to improve the project.

Thank you for your interest in contributing to `chat-gpt-lib-rs`!

## Example project
There is an interesting project [teachlead](https://crates.io/crates/techlead) now utilizing this project.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
