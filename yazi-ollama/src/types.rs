use serde::{Deserialize, Serialize};

// --- Generate ---

#[derive(Debug, Serialize)]
pub struct GenerateRequest {
	pub model:  String,
	pub prompt: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub suffix: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub system: Option<String>,
	pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
	pub model:    String,
	pub response: String,
	pub done:     bool,
}

// --- Chat ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
	pub role:    String,
	pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
	pub model:    String,
	pub messages: Vec<ChatMessage>,
	pub stream:   bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
	pub model:   String,
	pub message: ChatMessage,
	pub done:    bool,
}

// --- Embed ---

#[derive(Debug, Serialize)]
pub struct EmbedRequest {
	pub model:  String,
	pub input:  Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbedResponse {
	pub model:      String,
	pub embeddings: Vec<Vec<f64>>,
}
