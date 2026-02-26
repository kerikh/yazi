use serde::Deserialize;
use yazi_codegen::DeserializeOver2;

#[derive(Debug, Deserialize, DeserializeOver2)]
pub struct Ai {
	/// Whether the Ollama integration is enabled.
	pub ollama_enabled:     bool,
	/// Base URL of the Ollama server.
	pub ollama_url:         String,
	/// Default model to use for generate/chat requests.
	pub ollama_model:       String,
	/// Default model to use for embedding requests.
	pub ollama_embed_model: String,
}

impl Ai {
	pub(crate) fn reshape(self) -> anyhow::Result<Self> { Ok(self) }
}
