[![Crates.io](https://img.shields.io/crates/v/chat-gpt-lib-rs.svg)](https://crates.io/crates/chat-gpt-lib-rs) [![Documentation](https://docs.rs/chat-gpt-lib-rs/badge.svg)](https://docs.rs/chat-gpt-lib-rs/) [![Codecov](https://codecov.io/github/arend-jan/chat-gpt-lib-rs/coverage.svg?branch=main)](https://codecov.io/gh/arend-jan/chat-gpt-lib-rs) [![Dependency status](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs/status.svg)](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs)

# chat-gpt-lib-rs

A **Rust** client library for the [OpenAI API](https://platform.openai.com/docs/api-reference).  
Supports multiple OpenAI endpoints, including **Chat**, **Completions**, **Embeddings**, **Models**, **Moderations**, **Files**, **Fine-tunes**, and more ([chat-gpt-lib-rs 0.6.4 - Docs.rs](https://docs.rs/crate/chat-gpt-lib-rs/latest/source/README.md#:~:text=A%20,time%20responses)). Built with an **async-first** design using [Tokio](https://tokio.rs/) and [Reqwest](https://crates.io/crates/reqwest), featuring robust error handling and SSE streaming for real-time responses ([chat-gpt-lib-rs 0.6.4 - Docs.rs](https://docs.rs/crate/chat-gpt-lib-rs/latest/source/README.md#:~:text=Supports%20multiple%20OpenAI%20endpoints%2C%20including,time%20responses)).

> **Important**: If you’re upgrading from **0.5.x** to **0.6.x**, note that this transition introduces **significant breaking changes**. The project has been extensively refactored, making it too complex for a simple migration guide. You will likely need to update function calls and data structures to align with the new design ([chat-gpt-lib-rs 0.6.4 - Docs.rs](https://docs.rs/crate/chat-gpt-lib-rs/latest/source/README.md#:~:text=%3E%20,rs%29%20for%20detailed%20guidance)). Refer to the updated [examples](examples/) folder or the [documentation](https://docs.rs/chat-gpt-lib-rs) for guidance.

## Table of Contents

1. [Features](#features)  
2. [Installation](#installation)  
3. [Quick Start](#quick-start)  
4. [API Highlights](#api-highlights)  
   - [Models](#models)  
   - [Completions](#completions)  
   - [Chat Completions](#chat-completions)  
   - [Embeddings](#embeddings)  
   - [Moderations](#moderations)  
   - [Files](#files)  
   - [Fine-Tunes](#fine-tunes)  
5. [Environment Variables & Configuration](#environment-variables--configuration)  
6. [Streaming (SSE)](#streaming-sse)  
7. [Example Projects](#example-projects)  
8. [Contributing](#contributing)  
9. [License](#license)  

## Features

- **Async-first** – Built on [Tokio](https://tokio.rs/) + [Reqwest](https://crates.io/crates/reqwest) for asynchronous networking.  
- **Complete Coverage of Major OpenAI API Endpoints** – Supports:  
  - Chat (with streaming SSE for partial responses)  
  - Completions  
  - Models (list and retrieve)  
  - Embeddings  
  - Moderations  
  - Files (upload, list, download, delete)  
  - Fine-Tunes (create, list, retrieve, cancel, events, delete models) ([chat-gpt-lib-rs 0.6.4 - Docs.rs](https://docs.rs/crate/chat-gpt-lib-rs/latest#:~:text=%2A%20Async,retrieve%2C%20cancel%2C%20events%2C%20delete%20models))  
- **TLS without OpenSSL** – Uses Rustls for TLS, avoiding system OpenSSL dependencies ([chat-gpt-lib-rs/README.md at main · Arend-Jan/chat-gpt-lib-rs · GitHub](https://github.com/Arend-Jan/chat-gpt-lib-rs/blob/main/README.md#:~:text=,avoids%20system%20dependencies%20like%20OpenSSL)).  
- **Robust Error Handling** – Custom [`OpenAIError`](https://docs.rs/chat-gpt-lib-rs/latest/chat_gpt_lib_rs/enum.OpenAIError.html) covers HTTP errors, API errors, and JSON deserialization issues.  
- **Strongly-Typed Requests/Responses** – Serde-powered structs for all request and response bodies.  
- **Configurable** – Builder pattern to customize timeouts, organization IDs, or base URLs for proxy/Azure endpoints (while defaulting to OpenAI’s API URL).

## Installation

Add the crate to your `Cargo.toml` dependencies:

```toml
[dependencies]
chat-gpt-lib-rs = ""   # Use the latest version from crates.io
tokio = { version = "1", features = ["full"] }   # required for async runtime
```

Then build your project with Cargo:

```bash
cargo build
```

*Note:* The library is asynchronous and requires a Tokio runtime (as shown above). Ensure you have an async executor (like Tokio) to run the async functions.

## Quick Start

Below is a minimal example using **Completions** (with a Tokio runtime):

```rust
use chat_gpt_lib_rs::{OpenAIClient, OpenAIError};
use chat_gpt_lib_rs::api_resources::completions::{create_completion, CreateCompletionRequest};

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Initialize the OpenAI client. If no API key is provided, it will use the OPENAI_API_KEY env variable.
    let client = OpenAIClient::new(None)?;

    // Prepare a request to generate a text completion.
    let request = CreateCompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: Some("Write a short advertisement for ice cream.".into()),
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    // Call the Completions API.
    let response = create_completion(&client, &request).await?;
    println!("Completion Response:\n{:?}", response);

    Ok(())
}
```

## API Highlights

Below are examples of how to use various API endpoints. All calls require an `OpenAIClient` (shown as `client`), which handles authentication and HTTP configuration.

### Models

```rust
use chat_gpt_lib_rs::api_resources::models;
// List available models:
let all_models = models::list_models(&client).await?;
println!("Available Models: {:?}", all_models.data);

// Retrieve details for a specific model:
let model_details = models::retrieve_model(&client, "text-davinci-003").await?;
println!("Model details: {:?}", model_details);
```

### Completions

```rust
use chat_gpt_lib_rs::api_resources::completions::{
    create_completion, CreateCompletionRequest
};

let req = CreateCompletionRequest {
    model: "text-davinci-003".to_string(),
    prompt: Some("Hello, world!".into()),
    max_tokens: Some(50),
    ..Default::default()
};

let resp = create_completion(&client, &req).await?;
println!("Completion:\n{:?}", resp);
```

### Chat Completions

```rust
use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion, CreateChatCompletionRequest, ChatMessage, ChatRole
};

let chat_req = CreateChatCompletionRequest {
    model: "gpt-3.5-turbo".into(),
    messages: vec![
        ChatMessage {
            role: ChatRole::System,
            content: "You are a helpful assistant.".to_string(),
            name: None,
        },
        ChatMessage {
            role: ChatRole::User,
            content: "Give me a fun fact about Rust.".to_string(),
            name: None,
        },
    ],
    max_tokens: Some(50),
    ..Default::default()
};

let response = create_chat_completion(&client, &chat_req).await?;
println!("Chat reply:\n{:?}", response);
```

### Embeddings

```rust
use chat_gpt_lib_rs::api_resources::embeddings::{
    create_embeddings, CreateEmbeddingsRequest, EmbeddingsInput
};

let emb_req = CreateEmbeddingsRequest {
    model: "text-embedding-ada-002".to_string(),
    input: EmbeddingsInput::String("Hello world!".to_string()),
    user: None,
};
let emb_res = create_embeddings(&client, &emb_req).await?;
println!("Embedding vector:\n{:?}", emb_res.data[0].embedding);
```

### Moderations

```rust
use chat_gpt_lib_rs::api_resources::moderations::{
    create_moderation, CreateModerationRequest, ModerationsInput
};

let mod_req = CreateModerationRequest {
    input: ModerationsInput::String("I hate you and want to harm you.".into()),
    model: None,
};
let mod_res = create_moderation(&client, &mod_req).await?;
println!("Moderation flagged?: {}", mod_res.results[0].flagged);
```

### Files

```rust
use chat_gpt_lib_rs::api_resources::files::{
    upload_file, list_files, retrieve_file_content, delete_file, UploadFilePurpose
};
use std::path::PathBuf;

let file_path = PathBuf::from("training_data.jsonl");
// Upload a file for fine-tuning:
let upload = upload_file(&client, &file_path, UploadFilePurpose::FineTune).await?;
println!("Uploaded file ID: {}", upload.id);

// List all files:
let all_files = list_files(&client).await?;
println!("All files: {:?}", all_files.data);

// Retrieve content of the uploaded file:
let content = retrieve_file_content(&client, &upload.id).await?;
println!("File content size: {} bytes", content.len());

// Delete the file:
delete_file(&client, &upload.id).await?;
println!("File deleted.");
```

### Fine-Tunes

```rust
use chat_gpt_lib_rs::api_resources::fine_tunes::{
    create_fine_tune, list_fine_tunes, retrieve_fine_tune, cancel_fine_tune,
    list_fine_tune_events, CreateFineTuneRequest
};

// Create a fine-tuning job:
let ft_req = CreateFineTuneRequest {
    training_file: "file-abc123".into(),
    model: Some("curie".to_string()),
    ..Default::default()
};
let job = create_fine_tune(&client, &ft_req).await?;
println!("Created fine-tune job: {}", job.id);

// List all fine-tune jobs:
let all_jobs = list_fine_tunes(&client).await?;
println!("All fine-tune jobs: {:?}", all_jobs.data);

// Get details or events for a specific fine-tune job:
let job_details = retrieve_fine_tune(&client, &job.id).await?;
println!("Fine-tune job status: {}", job_details.status);

let events = list_fine_tune_events(&client, &job.id).await?;
println!("Fine-tune events: {:?}", events.data);

// Cancel an ongoing fine-tune job (if needed):
cancel_fine_tune(&client, &job.id).await?;
```

## Environment Variables & Configuration

By default, the library reads your OpenAI API key from the `OPENAI_API_KEY` environment variable ([chat-gpt-lib-rs/README.md at main · Arend-Jan/chat-gpt-lib-rs · GitHub](https://github.com/Arend-Jan/chat-gpt-lib-rs/blob/main/README.md#:~:text=By%20default%2C%20the%20library%20reads,OPENAI_API_KEY)). Set it in your shell or in a `.env` file (you can use the [dotenvy](https://crates.io/crates/dotenvy) crate to load it):

```bash
export OPENAI_API_KEY="sk-YourAPIKeyHere"
```

Alternatively, you can provide the API key directly in code:

```rust
let client = OpenAIClient::new(Some("sk-your-api-key".to_string()))?;
```

For advanced configuration, use the **builder** to customize settings:

```rust
use chat_gpt_lib_rs::OpenAIClient;
use std::time::Duration;

let client = OpenAIClient::builder()
    .with_api_key("sk-your-api-key")            // explicitly set API key (otherwise reads OPENAI_API_KEY)
    .with_organization("org-your-org-id")       // set your OpenAI organization ID (if applicable)
    .with_timeout(Duration::from_secs(30))      // custom request timeout for API calls
    // .with_base_url("https://api.openai.com/v1/")  // optionally override base URL for OpenAI API (or Azure proxy)
    .build()
    .unwrap();
``` 

If not specified, the client will default to OpenAI’s public API endpoint and no organization ID.

## Streaming (SSE)

For real-time partial responses, you can request streaming results. Set `stream: true` in your request to the Chat or Completions API, which will return a stream that yields incremental updates ([chat-gpt-lib-rs/README.md at main · Arend-Jan/chat-gpt-lib-rs · GitHub](https://github.com/Arend-Jan/chat-gpt-lib-rs/blob/main/README.md#:~:text=Streaming%20)). You can then process the stream as each chunk arrives:

```rust
use futures_util::StreamExt;        // for StreamExt::next()
use std::io::{self, Write};         // for flushing stdout
use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion_stream, CreateChatCompletionRequest, ChatMessage, ChatRole,
};
use chat_gpt_lib_rs::{OpenAIClient, OpenAIError};

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    let client = OpenAIClient::new(None)?;
    let request = CreateChatCompletionRequest {
        model: "gpt-3.5-turbo".into(),
        messages: vec![
            ChatMessage { role: ChatRole::System, content: "You are a helpful assistant.".into(), name: None },
            ChatMessage { role: ChatRole::User, content: "Tell me a joke.".into(), name: None },
        ],
        stream: Some(true),  // enable streaming
        ..Default::default()
    };

    println!("Streaming response:\n");
    let mut stream = create_chat_completion_stream(&client, &request).await?;
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Each chunk contains a delta with part of the message
                if let Some(choice) = chunk.choices.first() {
                    if let Some(text) = &choice.delta.content {
                        print!("{}", text);
                        io::stdout().flush().unwrap();
                    }
                }
            }
            Err(e) => eprintln!("Stream error: {:?}", e),
        }
    }
    println!("\n[Done]");
    Ok(())
}
```

In the above example, as the stream yields each `chunk` of the completion, we immediately print the `delta.content` (partial message) without waiting for the full response. This provides a real-time typing effect until the stream ends with a final `[DONE]` message.

## Example Projects

Check out the `examples/` directory in this repository for more comprehensive examples, including a CLI chat demo and usage of streaming. These examples demonstrate common patterns and can be used as a starting point for your own applications.

**Third-Party Usage:** The [`techlead`](https://crates.io/crates/techlead) crate (an AI chat CLI) uses **chat-gpt-lib-rs** for its OpenAI interactions ([chat-gpt-lib-rs/README.md at main · Arend-Jan/chat-gpt-lib-rs · GitHub](https://github.com/Arend-Jan/chat-gpt-lib-rs/blob/main/README.md#:~:text=Third)). You can refer to it as a real-world example of integrating this library.

## Contributing

Contributions and feedback are welcome! To get started:

1. Fork the repository and clone your fork.  
2. Create a new branch for your feature or fix.  
3. Implement your changes, with tests if applicable.  
4. Run tests to ensure nothing is broken (`cargo test`).  
5. Open a pull request describing your changes.

Given that the 0.6.x release was a major refactor, much of the code has changed from earlier versions. If you are updating older code, please refer to the new examples and documentation for the updated usage patterns ([chat-gpt-lib-rs/README.md at main · Arend-Jan/chat-gpt-lib-rs · GitHub](https://github.com/Arend-Jan/chat-gpt-lib-rs/blob/main/README.md#:~:text=4,describing%20the%20changes)). This will help in understanding how to migrate to the latest API.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
