# streamlit-pandasai-ollama/app.py
import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

st.set_page_config(
    page_title="Chat with your CSV — Local LLM (PandasAI + Ollama)",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Chat with your CSV — Local LLM (PandasAI + Ollama)")
st.caption("Private & local: Streamlit + PandasAI v3 + LiteLLM + Ollama")

# Sidebar: configure Ollama + model
st.sidebar.header("⚙️ Settings")
ollama_base = st.sidebar.text_input("Ollama base URL", value="http://localhost:11434")
model_choice = st.sidebar.selectbox(
    "Local model",
    ["ollama/phi3.5", "ollama/qwen2.5:0.5b", "ollama/qwen2.5:1.5b", "ollama/llama3.2:1b"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
max_output_tokens = st.sidebar.slider("Max output tokens", 256, 4096, 1024, 64)
stream_ctx_includes_df = st.sidebar.checkbox(
    "Include CSV context in Streaming mode", value=True,
    help="Streaming mode does not execute code; this only adds columns and a small sample to the prompt."
)

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

# Wire PandasAI to LiteLLM (which talks to Ollama) — non-stream LLM for PandasAI
llm = LiteLLM(model=model_choice, api_base=ollama_base)
pai.config.set({"llm": llm, "temperature": temperature, "max_output_tokens": max_output_tokens})

# Chat state (separate histories for each tab)
if "pandas_history" not in st.session_state:
    st.session_state.pandas_history = []
if "stream_history" not in st.session_state:
    st.session_state.stream_history = []

# Tabs: PandasAI vs Streaming
tab_df, tab_stream = st.tabs(["🔢 PandasAI mode (Data-aware)", "⏱️ Streaming mode (raw LLM)"])

# --------------------------
# Tab 1: PandasAI mode
# --------------------------
with tab_df:
    if uploaded_file is None:
        st.info("Upload a CSV to start chatting with your data.")
    else:
        df = pai.read_csv(uploaded_file)  # DataFrame with .chat()
        st.subheader("Data preview")
        try:
            st.dataframe(df.head(10))
        except Exception:
            # Fallback: if PandasAI read fails, use pandas then wrap in pai.DataFrame
            fallback_df = pd.read_csv(uploaded_file)
            st.dataframe(fallback_df.head(10))
            df = pai.DataFrame(fallback_df)

        st.divider()
        st.subheader("Ask a question about your CSV (PandasAI)")
        user_msg_df = st.chat_input("Type a question about your CSV…", key="pandas_chat_input")

        if user_msg_df:
            st.session_state.pandas_history.append(("user", user_msg_df))
            # Use a status panel to show progress (PandasAI is blocking; not token streaming)
            with st.status("Working with your local model…", expanded=True) as status:
                st.write("🔧 Building prompt & planning…")
                try:
                    response = df.chat(user_msg_df)  # Blocking call: returns final result (text/plot/figure)
                    st.write("🧠 Model completed reasoning & code generation.")
                    st.write("▶️ Executed code / generated outputs (if any).")
                    status.update(label="Done!", state="complete")
                except Exception as e:
                    response = (
                        f"Error: {e}\n\nCheck that Ollama is running at {ollama_base} "
                        f"and '{model_choice}' is pulled."
                    )
                    status.update(label="Failed", state="error")
            st.session_state.pandas_history.append(("assistant", response))

        for role, content in st.session_state.pandas_history:
            with st.chat_message(role):
                # If content is a matplotlib figure, render it; else show text
                if hasattr(content, "__str__") and not hasattr(content, "savefig"):
                    st.write(str(content))
                else:
                    try:
                        st.pyplot(content)  # render matplotlib figures
                    except Exception:
                        st.write(content)

        st.divider()
        st.markdown("💡 Try: `Plot monthly sales by region`, `Find rows with missing price`, `Top 10 customers by orders`.")

# --------------------------
# Tab 2: Streaming mode (raw LLM)
# --------------------------
with tab_stream:
    st.write("This tab **streams tokens in real time** (word-by-word).")
    st.write(
        "It does **not** execute code or generate plots. "
        "Optionally, it can include CSV context (columns + a small sample) to help the model answer."
    )

    # Build optional CSV context prompt
    csv_context_prompt = ""
    if uploaded_file is not None and stream_ctx_includes_df:
        try:
            preview_df = pd.read_csv(uploaded_file)
            cols = list(preview_df.columns)
            sample_rows = preview_df.head(5).to_dict(orient="records")
            csv_context_prompt = (
                "CSV Context:\n"
                f"- Columns: {cols}\n"
                f"- Sample rows (first 5): {sample_rows}\n\n"
                "You may reference this context in your answer, "
                "but DO NOT claim you executed code or performed calculations on the full dataset."
            )
        except Exception:
            csv_context_prompt = (
                "A CSV was uploaded, but we couldn't read a preview for context. "
                "Proceed without data execution."
            )

    # Display prior messages for the streaming chat
    for role, content in st.session_state.stream_history:
        with st.chat_message(role):
            st.write(content)

    # Streaming chat input
    user_msg_stream = st.chat_input("Ask anything… (streams word-by-word)", key="stream_chat_input")

    if user_msg_stream:
        # Persist user's message
        st.session_state.stream_history.append(("user", user_msg_stream))

        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_msg_stream)

        # Prepare messages for the LLM (system + history + new user message)
        messages = []
        system_text = (
            "You are a helpful local assistant. "
            "Respond concisely and stream your answer as tokens. "
            "If CSV context is provided below, you may reference it, "
            "but DO NOT claim you executed code or generated plots.\n\n"
        )
        if csv_context_prompt:
            system_text += csv_context_prompt
        messages.append({"role": "system", "content": system_text})

        # Add previous history from streaming mode
        for role, content in st.session_state.stream_history:
            if role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})

        # Create a streaming LLM
        # Note: LiteLLM's streaming API may yield plain text chunks or dicts; we handle both.
        llm_stream = LiteLLM(model=model_choice, api_base=ollama_base, stream=True)

        # Stream tokens to the UI incrementally
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""

            try:
                for chunk in llm_stream.stream_chat(messages=messages):
                    # chunk may be a dict with 'content' or a bare text piece
                    if isinstance(chunk, dict):
                        piece = chunk.get("content", "")
                    else:
                        piece = str(chunk)
                    full_text += piece
                    # Update the UI progressively
                    placeholder.markdown(full_text)
            except Exception as e:
                full_text = f"Streaming error: {e}\n\nCheck Ollama at {ollama_base} and ensure '{model_choice}' is pulled."

            # Persist assistant's streamed completion
            st.session_state.stream_history.append(("assistant", full_text))
