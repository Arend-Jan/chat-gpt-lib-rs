//! The `config` module provides functionality for configuring and creating the [`OpenAIClient`],
//! including handling API keys, organization IDs, timeouts, and base URLs.
//
//! # Overview
//!
//! This module exposes the [`OpenAIClient`] struct, which is your main entry point for interacting
//! with the OpenAI API. It provides a builder-pattern (`ClientBuilder`) for customizing various
//! aspects of the client configuration, such as the API key, organization ID, timeouts, and so on.
//
//! # Usage
//!
//! ```rust
//! use chat_gpt_lib_rs::OpenAIClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!      // Load environment variables from a .env file, if present (optional).
//!      dotenvy::dotenv().ok();
//!
//!     // Example 1: Use environment variable `OPENAI_API_KEY`.
//!     let client = OpenAIClient::new(None)?;
//!
//!     // Example 2: Use a builder pattern to set more configuration.
//!     let client_with_org = OpenAIClient::builder()
//!         .with_api_key("sk-...YourKey...")
//!         .with_organization("org-MyOrganization")
//!         .with_timeout(std::time::Duration::from_secs(30))
//!         .build()?;
//!
//!     // Use `client` or `client_with_org` to make API requests...
//!
//!     Ok(())
//! }
//! ```

use std::env;
use std::time::Duration;

use reqwest::{Client, ClientBuilder as HttpClientBuilder};

use crate::error::OpenAIError;

/// The default base URL for the OpenAI API.
///
/// You can override this in the builder if needed (e.g., for proxies or mock servers).
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1/";

/// A client for interacting with the OpenAI API.
///
/// This struct holds the configuration (e.g., API key, organization ID, base URL) and
/// an underlying [`reqwest::Client`] for making HTTP requests. Typically, you'll create an
/// `OpenAIClient` using:
/// 1) The [`OpenAIClient::new`] method, which optionally reads the API key from an environment variable, or
/// 2) The builder pattern via [`OpenAIClient::builder`].
#[derive(Clone, Debug)]
pub struct OpenAIClient {
    /// The full base URL used for OpenAI endpoints (e.g. "https://api.openai.com/v1/").
    base_url: String,
    /// The API key used for authentication (e.g., "sk-...").
    api_key: String,
    /// Optional organization ID, if applicable to your account.
    organization: Option<String>,
    /// The underlying HTTP client from `reqwest`, configured with timeouts, TLS, etc.
    pub(crate) http_client: Client,
}

impl OpenAIClient {
    /// Creates a new `OpenAIClient` using the provided API key, or reads it from the
    /// `OPENAI_API_KEY` environment variable if `api_key` is `None`.
    ///
    /// # Errors
    ///
    /// Returns an [`OpenAIError`] if no API key can be found in the given argument or
    /// the environment variable.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use chat_gpt_lib_rs::OpenAIClient;
    /// // load environment variables from a .env file, if present (optional).
    /// dotenvy::dotenv().ok();
    ///
    /// // Reads `OPENAI_API_KEY` from the environment.
    /// let client = OpenAIClient::new(None).unwrap();
    ///
    /// // Provide an explicit API key.
    /// let client = OpenAIClient::new(Some("sk-...".to_string())).unwrap();
    /// ```
    pub fn new(api_key: Option<String>) -> Result<Self, OpenAIError> {
        let key = match api_key {
            Some(k) => k,
            None => env::var("OPENAI_API_KEY")
                .map_err(|_| OpenAIError::ConfigError("Missing API key".to_string()))?,
        };

        let http_client = HttpClientBuilder::new()
            .build()
            .map_err(|e| OpenAIError::ConfigError(e.to_string()))?;

        Ok(Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: key,
            organization: None,
            http_client,
        })
    }

    /// Returns a new [`ClientBuilder`] to configure and build an `OpenAIClient`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use chat_gpt_lib_rs::OpenAIClient;
    /// let client = OpenAIClient::builder()
    ///     .with_api_key("sk-EXAMPLE")
    ///     .with_organization("org-EXAMPLE")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Returns the current base URL as a string slice.
    ///
    /// Useful if you need to verify or debug the client's configuration.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the API key as a string slice.
    ///
    /// For security reasons, you might not want to expose this in production logs.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Returns the optional organization ID, if it was set.
    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }
}

/// A builder for [`OpenAIClient`] that follows the builder pattern.
///
/// # Examples
///
/// ```rust
/// use chat_gpt_lib_rs::OpenAIClient;
/// use std::time::Duration;
///
/// let client = OpenAIClient::builder()
///     .with_api_key("sk-...YourKey...")
///     .with_organization("org-MyOrganization")
///     .with_timeout(Duration::from_secs(30))
///     .build()
///     .unwrap();
/// ```
#[derive(Default, Debug)]
pub struct ClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    organization: Option<String>,
    timeout: Option<Duration>,
}

impl ClientBuilder {
    /// Sets a custom base URL for all OpenAI requests.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chat_gpt_lib_rs::OpenAIClient;
    /// let client = OpenAIClient::builder()
    ///     .with_base_url("https://custom-openai-proxy.example.com/v1/")
    ///     .with_api_key("sk-EXAMPLE")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = Some(url.to_string());
        self
    }

    /// Sets the API key explicitly. If not provided, the client will attempt to
    /// read from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chat_gpt_lib_rs::OpenAIClient;
    /// let client = OpenAIClient::builder()
    ///     .with_api_key("sk-EXAMPLE")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_api_key(mut self, key: &str) -> Self {
        self.api_key = Some(key.to_string());
        self
    }

    /// Sets the organization ID for the client. Some accounts or requests
    /// require specifying an organization ID.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chat_gpt_lib_rs::OpenAIClient;
    /// let client = OpenAIClient::builder()
    ///     .with_api_key("sk-EXAMPLE")
    ///     .with_organization("org-EXAMPLE")
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_organization(mut self, org: &str) -> Self {
        self.organization = Some(org.to_string());
        self
    }

    /// Sets a timeout for all HTTP requests made by this client.
    /// If not specified, the timeout behavior of the underlying
    /// [`reqwest::Client`] defaults are used.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chat_gpt_lib_rs::OpenAIClient;
    /// # use std::time::Duration;
    /// let client = OpenAIClient::builder()
    ///     .with_api_key("sk-EXAMPLE")
    ///     .with_timeout(Duration::from_secs(30))
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Builds the [`OpenAIClient`] using the specified configuration.
    ///
    /// If the API key is not set through `with_api_key`, it attempts to read from
    /// the `OPENAI_API_KEY` environment variable. If no key is found, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns an [`OpenAIError`] if no API key is provided or discovered in the environment,
    /// or if building the underlying HTTP client fails.
    pub fn build(self) -> Result<OpenAIClient, OpenAIError> {
        // Determine the base URL
        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());

        // Determine the API key
        let api_key = match self.api_key {
            Some(k) => k,
            None => env::var("OPENAI_API_KEY")
                .map_err(|_| OpenAIError::ConfigError("Missing API key".to_string()))?,
        };

        let organization = self.organization;

        // Build the reqwest Client with optional timeout
        let mut http_client_builder = HttpClientBuilder::new();
        if let Some(to) = self.timeout {
            http_client_builder = http_client_builder.timeout(to);
        }

        // Build the reqwest client
        let http_client = http_client_builder
            .build()
            .map_err(|e| OpenAIError::ConfigError(e.to_string()))?;

        Ok(OpenAIClient {
            base_url,
            api_key,
            organization,
            http_client,
        })
    }
}
