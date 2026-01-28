# 📊 Chat-with-CSV (Local LLM) — Streamlit + PandasAI + Ollama

This project is a **fully local** Streamlit app that lets you **chat with a CSV** using:
- **PandasAI mode** → actually *data-aware* (can plan/execute pandas logic + charts)
- **Streaming mode** → token-by-token *raw LLM vibes* 

No cloud. No API keys. Just your laptop doing honest work. 💻🔥

---

## ✨ Features

### 🔢 PandasAI mode (Data-aware)
- Upload a CSV and ask questions like:
  - “Top 10 customers by orders”
  - “Find rows with missing price”
  - “Plot monthly sales by region”
- Can return **text answers** and sometimes **plots**
- Shows a progress panel so you know it’s alive

### ⏱️ Streaming mode (raw LLM)
- Streams words live like a dramatic narrator 🎭
- **Does NOT execute code**
- Optional **CSV context injection** (columns + first 5 rows) to help the model answer
- Explicitly avoids pretending it computed on the full dataset ✅

---

## 🧠 Tech Stack

- **Streamlit** (UI)
- **Pandas** (CSV handling)
- **PandasAI v3** (data-aware chat + tools)
- **LiteLLM adapter** (`pandasai_litellm`) (bridges PandasAI ↔ Ollama)
- **Ollama** (local LLM runtime)
- Models supported in-app:
  - `phi3.5`
  - `qwen2.5:0.5b`
  - `qwen2.5:1.5b`
  - `llama3.2:1b`

---

## 📁 Repo Structure

```text
streamlit-pandasai-ollama/
├── .venv/                # Virtual environment (ignored)
├── exports/              # Charts / artifacts (if generated)
├── .gitignore            # Ignore logs + venv stuff
├── app.py                # The Streamlit app
├── pandasai.log          # PandasAI logs (ignored)
├── requirements.txt      # Dependencies

