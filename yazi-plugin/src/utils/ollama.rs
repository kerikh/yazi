use mlua::{ExternalError, ExternalResult, Function, Lua, Table, Value};
use yazi_config::YAZI;
use yazi_ollama::{ChatMessage, OllamaClient};

use super::Utils;

impl Utils {
	pub(super) fn ollama_generate(lua: &Lua) -> mlua::Result<Function> {
		lua.create_async_function(|lua, (prompt, opts): (String, Option<Table>)| async move {
			let ai = &YAZI.ai;
			if !ai.ollama_enabled {
				return Err("Ollama integration is disabled. Set `[ai].ollama_enabled = true` in yazi.toml".into_lua_err());
			}

			let client = OllamaClient::new(&ai.ollama_url).into_lua_err()?;

			let (model, system) = if let Some(ref t) = opts {
				let model: Option<String> = t.get("model")?;
				let system: Option<String> = t.get("system")?;
				(model.unwrap_or_else(|| ai.ollama_model.clone()), system)
			} else {
				(ai.ollama_model.clone(), None)
			};

			let resp = client.generate(model, prompt, system).await.into_lua_err()?;

			let tbl = lua.create_table()?;
			tbl.set("model", resp.model)?;
			tbl.set("response", resp.response)?;
			tbl.set("done", resp.done)?;
			Ok(Value::Table(tbl))
		})
	}

	pub(super) fn ollama_chat(lua: &Lua) -> mlua::Result<Function> {
		lua.create_async_function(|lua, (messages, opts): (Table, Option<Table>)| async move {
			let ai = &YAZI.ai;
			if !ai.ollama_enabled {
				return Err("Ollama integration is disabled. Set `[ai].ollama_enabled = true` in yazi.toml".into_lua_err());
			}

			let client = OllamaClient::new(&ai.ollama_url).into_lua_err()?;

			let model = if let Some(ref t) = opts {
				let m: Option<String> = t.get("model")?;
				m.unwrap_or_else(|| ai.ollama_model.clone())
			} else {
				ai.ollama_model.clone()
			};

			let mut msgs: Vec<ChatMessage> = Vec::new();
			for pair in messages.sequence_values::<Table>() {
				let t = pair?;
				msgs.push(ChatMessage { role: t.get("role")?, content: t.get("content")? });
			}

			let resp = client.chat(model, msgs).await.into_lua_err()?;

			let tbl = lua.create_table()?;
			tbl.set("model", resp.model)?;
			let msg_tbl = lua.create_table()?;
			msg_tbl.set("role", resp.message.role)?;
			msg_tbl.set("content", resp.message.content)?;
			tbl.set("message", msg_tbl)?;
			tbl.set("done", resp.done)?;
			Ok(Value::Table(tbl))
		})
	}

	pub(super) fn ollama_embed(lua: &Lua) -> mlua::Result<Function> {
		lua.create_async_function(|lua, (inputs, opts): (Value, Option<Table>)| async move {
			let ai = &YAZI.ai;
			if !ai.ollama_enabled {
				return Err("Ollama integration is disabled. Set `[ai].ollama_enabled = true` in yazi.toml".into_lua_err());
			}

			let client = OllamaClient::new(&ai.ollama_url).into_lua_err()?;

			let model = if let Some(ref t) = opts {
				let m: Option<String> = t.get("model")?;
				m.unwrap_or_else(|| ai.ollama_embed_model.clone())
			} else {
				ai.ollama_embed_model.clone()
			};

			let texts: Vec<String> = match inputs {
				Value::String(s) => vec![s.to_str()?.to_owned()],
				Value::Table(t) => t.sequence_values::<String>().collect::<mlua::Result<_>>()?,
				other => {
					return Err(
						format!("expected string or table, got {}", other.type_name()).into_lua_err(),
					);
				}
			};

			let resp = client.embed(model, texts).await.into_lua_err()?;

			let tbl = lua.create_table()?;
			tbl.set("model", resp.model)?;
			let embeddings_tbl = lua.create_table()?;
			for (i, vec) in resp.embeddings.iter().enumerate() {
				let row = lua.create_table()?;
				for (j, &v) in vec.iter().enumerate() {
					row.set(j + 1, v)?;
				}
				embeddings_tbl.set(i + 1, row)?;
			}
			tbl.set("embeddings", embeddings_tbl)?;
			Ok(Value::Table(tbl))
		})
	}
}
