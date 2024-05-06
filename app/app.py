import streamlit as st
from pipeline import (load_embeddings,
                      load_text_chunks,
                      search_text, load_llm,
                      generate_response, get_device, load_encoder)


@st.cache_resource(show_spinner=False)
def load_resources():
    with st.spinner("Loading resources..."):
        device = get_device()
        embeddings = load_embeddings()
        text_chunks = load_text_chunks()

    with st.spinner("Loading encoder..."):
        encoder = load_encoder()

    with st.spinner("Loading LLM..."):
        llm, tokenizer = load_llm()
    return device, embeddings, text_chunks, encoder, llm, tokenizer


def main():
    st.title("Chatbot with RAG")

    device, embeddings, text_chunks, encoder, llm, tokenizer = load_resources()

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(st.session_state.messages[-1]["content"])

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks, titles, scores = search_text(query, embeddings, text_chunks, encoder, device)
                response = generate_response(query, llm, tokenizer, device, chunks)

                response = response.split("<start_of_turn>model")[-1].strip()
                response = response.replace("<eos>", "")
            st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
