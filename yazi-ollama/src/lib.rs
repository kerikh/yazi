mod client;
mod types;

pub use client::OllamaClient;
pub use types::{ChatMessage, ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse};
