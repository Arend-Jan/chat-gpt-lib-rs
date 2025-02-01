[![Crates.io](https://img.shields.io/crates/v/chat-gpt-lib-rs.svg)](https://crates.io/crates/chat-gpt-lib-rs)
[![Documentation](https://docs.rs/chat-gpt-lib-rs/badge.svg)](https://docs.rs/chat-gpt-lib-rs/)
[![Codecov](https://codecov.io/github/arend-jan/chat-gpt-lib-rs/coverage.svg?branch=main)](https://codecov.io/gh/arend-jan/chat-gpt-lib-rs)
[![Dependency status](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs/status.svg)](https://deps.rs/repo/github/arend-jan/chat-gpt-lib-rs)

# chat-gpt-lib-rs

A **Rust** client library for the [OpenAI API](https://platform.openai.com/docs/api-reference).  
Supports multiple OpenAI endpoints, including **Chat**, **Completions**, **Embeddings**, **Models**, **Moderations**, **Files**, **Fine-tunes**, and more. Built with **async**-first design using [Tokio](https://tokio.rs/) and [Reqwest](https://crates.io/crates/reqwest), featuring robust error handling and SSE streaming for real-time responses.

> **Important**: If you’re upgrading from **0.5.x** to **0.6.x**, please note that this transition introduces **significant, breaking changes**. The project has been extensively refactored, making it too complex for a straightforward migration guide. You will likely need to update function calls and data structures to align with the new design. Refer to the updated [examples](examples/) folder or the [documentation](https://docs.rs/chat-gpt-lib-rs) for detailed guidance.

---

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
5. [Environment Variables](#environment-variables)  
6. [Streaming (SSE)](#streaming-sse)  
7. [Example Projects](#example-projects)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Features

- **Async-first**: Built on [Tokio](https://tokio.rs/) + [Reqwest](https://crates.io/crates/reqwest).  
- **Complete Coverage** of major OpenAI API endpoints:
  - **Chat** (with streaming SSE for partial responses)
  - **Completions**  
  - **Models** (list and retrieve)
  - **Embeddings**  
  - **Moderations**  
  - **Files** (upload, list, download, delete)
  - **Fine-Tunes** (create, list, retrieve, cancel, events, delete models)  
- **Rustls** for TLS: avoids system dependencies like OpenSSL.  
- **Thorough Error Handling** with custom [`OpenAIError`](https://docs.rs/chat-gpt-lib-rs/latest/chat_gpt_lib_rs/error/enum.OpenAIError.html).  
- **Typed** request/response structures (Serde-based).  
- **Extensive Documentation** and usage examples, including SSE streaming.

---

## Installation

In your `Cargo.toml`, under `[dependencies]`:

```toml
chat-gpt-lib-rs = "x.y.z"  # Replace x.y.z with the latest version
```

Then build your project:

```bash
cargo build
```

---

## Quick Start

Below is a **minimal** example using **Completions**:

```rust
use chat_gpt_lib_rs::{OpenAIClient, OpenAIError};
use chat_gpt_lib_rs::api_resources::completions::{create_completion, CreateCompletionRequest};

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Pass your API key directly or rely on the OPENAI_API_KEY environment variable
    let client = OpenAIClient::new(None)?;

    // Prepare a request to generate text completions
    let request = CreateCompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: Some("Write a short advertisement for ice cream.".into()),
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..Default::default()
    };

    // Call the Completions API
    let response = create_completion(&client, &request).await?;
    println!("Completion Response:\n{:?}", response);

    Ok(())
}
```

---

## API Highlights

### Models

```rust
use chat_gpt_lib_rs::api_resources::models;

let all_models = models::list_models(&client).await?;
println!("Available Models: {:?}", all_models);

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
println!("Embedding:\n{:?}", emb_res.data[0].embedding);
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
println!("Moderation result:\n{:?}", mod_res.results[0].flagged);
```

### Files

```rust
use chat_gpt_lib_rs::api_resources::files::{
    upload_file, list_files, retrieve_file_content, delete_file,
    UploadFilePurpose
};
use std::path::PathBuf;

let file_path = PathBuf::from("training_data.jsonl");
let upload = upload_file(&client, &file_path, UploadFilePurpose::FineTune).await?;
println!("Uploaded file ID: {}", upload.id);

let all_files = list_files(&client).await?;
println!("All files: {:?}", all_files.data);

let content = retrieve_file_content(&client, &upload.id).await?;
println!("File content size: {}", content.len());

delete_file(&client, &upload.id).await?;
```

### Fine-Tunes

```rust
use chat_gpt_lib_rs::api_resources::fine_tunes::{
    create_fine_tune, list_fine_tunes, CreateFineTuneRequest
};

let ft_req = CreateFineTuneRequest {
    training_file: "file-abc123".into(),
    model: Some("curie".to_string()),
    ..Default::default()
};
let job = create_fine_tune(&client, &ft_req).await?;
println!("Created fine-tune job: {}", job.id);

let all_jobs = list_fine_tunes(&client).await?;
println!("All fine-tune jobs: {:?}", all_jobs.data);
```

---

## Environment Variables

By default, the library reads your OpenAI API key from **`OPENAI_API_KEY`**:

```bash
export OPENAI_API_KEY="sk-xxx"
```

Or use a `.env` file with [dotenvy](https://crates.io/crates/dotenvy).

Alternatively, provide a key directly:

```rust
let client = OpenAIClient::new(Some("sk-your-key".to_string()))?;
```

---

## Streaming (SSE)

For **real-time** partial responses, pass `stream = true` to Chat or Completions endpoints and process the resulting stream:

```rust
use futures_util::StreamExt; // Import the extension trait for streams
use std::io::{self, Write};  // For flushing stdout
use chat_gpt_lib_rs::api_resources::chat::{
    create_chat_completion_stream, CreateChatCompletionRequest, ChatMessage, ChatRole,
};
use chat_gpt_lib_rs::OpenAIClient;
use chat_gpt_lib_rs::error::OpenAIError;

#[tokio::main]
async fn main() -> Result<(), OpenAIError> {
    // Create a new OpenAI client (API key is read from the environment, e.g. OPENAI_API_KEY)
    let client = OpenAIClient::new(None)?;

    // Build a chat request with a system prompt and a user message.
    // Note: Setting `stream: Some(true)` enables streaming responses.
    let request = CreateChatCompletionRequest {
        model: "gpt-3.5-turbo".into(),
        messages: vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a helpful assistant.".to_string(),
                name: None,
            },
            ChatMessage {
                role: ChatRole::User,
                content: "Tell me a joke.".to_string(),
                name: None,
            },
        ],
        stream: Some(true),
        ..Default::default()
    };

    println!("Sending chat streaming request...");

    // Retrieve the streaming response from the API.
    // Each item in the stream is a partial update (chunk) containing a delta.
    let mut stream = create_chat_completion_stream(&client, &request).await?;
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                // Each chunk contains a list of choices.
                // For streaming responses, the incremental text is in the `delta.content` field.
                if let Some(choice) = chunk.choices.first() {
                    if let Some(text) = &choice.delta.content {
                        // Print the incremental text without a newline and flush stdout immediately.
                        print!("{}", text);
                        io::stdout().flush().unwrap();
                    }
                }
            }
            Err(e) => eprintln!("Stream error: {:?}", e),
        }
    }
    println!("\nDone streaming.");
    Ok(())
}
```

---

## Example Projects

Check the `examples/` folder for CLI chat demos and more.

**Third-Party Usage**:  
- [techlead](https://crates.io/crates/techlead) uses this library for advanced AI-driven chat interactions.

---

## Contributing

We welcome contributions and feedback! To get started:

1. **Fork** this repository and clone your fork locally.  
2. **Create a branch** for your changes or fixes.  
3. **Make & test** your changes.  
4. **Submit a pull request** describing the changes.

Because this release is a **major refactor**, please note that much of the code has changed. If you’re updating older code, see the new examples and docs for updated usage patterns.

---

## License

Licensed under the **Apache License 2.0**—see [LICENSE](LICENSE) for full details.

---

**Breaking Changes Note**:  
Due to the **extensive** updates, we do **not** provide a direct migration guide. You may need to adapt your existing code to updated function signatures and data structures. Consult the new documentation and examples to get started quickly.
