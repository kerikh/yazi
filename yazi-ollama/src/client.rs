use anyhow::{Context, Result};
use reqwest::Client;

use crate::types::{
	ChatMessage, ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, GenerateRequest,
	GenerateResponse,
};

/// Client for communicating with a local Ollama server.
///
/// By default connects to `http://localhost:11434`.
#[derive(Clone, Debug)]
pub struct OllamaClient {
	client:   Client,
	base_url: String,
}

impl OllamaClient {
	/// Create a new client pointing at `base_url` (e.g. `"http://localhost:11434"`).
	pub fn new(base_url: impl Into<String>) -> Result<Self> {
		Ok(Self { client: Client::builder().build()?, base_url: base_url.into() })
	}

	/// Send a single-turn generate request and return the full response text.
	pub async fn generate(
		&self,
		model: impl Into<String>,
		prompt: impl Into<String>,
		system: Option<String>,
	) -> Result<GenerateResponse> {
		let req = GenerateRequest {
			model:  model.into(),
			prompt: prompt.into(),
			suffix: None,
			system,
			stream: false,
		};
		self.client
			.post(format!("{}/api/generate", self.base_url))
			.json(&req)
			.send()
			.await
			.context("Failed to reach Ollama server")?
			.json::<GenerateResponse>()
			.await
			.context("Failed to parse Ollama generate response")
	}

	/// Send a chat request with a history of messages and return the assistant reply.
	pub async fn chat(
		&self,
		model: impl Into<String>,
		messages: Vec<ChatMessage>,
	) -> Result<ChatResponse> {
		let req = ChatRequest { model: model.into(), messages, stream: false };
		self.client
			.post(format!("{}/api/chat", self.base_url))
			.json(&req)
			.send()
			.await
			.context("Failed to reach Ollama server")?
			.json::<ChatResponse>()
			.await
			.context("Failed to parse Ollama chat response")
	}

	/// Request embedding vectors for one or more text inputs.
	pub async fn embed(
		&self,
		model: impl Into<String>,
		inputs: Vec<String>,
	) -> Result<EmbedResponse> {
		let req = EmbedRequest { model: model.into(), input: inputs };
		self.client
			.post(format!("{}/api/embed", self.base_url))
			.json(&req)
			.send()
			.await
			.context("Failed to reach Ollama server")?
			.json::<EmbedResponse>()
			.await
			.context("Failed to parse Ollama embed response")
	}
}
