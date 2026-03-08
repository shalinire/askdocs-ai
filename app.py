import streamlit as st
import tempfile
import time

from ingest import process_pdf, reset_database
from retrieval import ask_question

st.set_page_config(
    page_title="AskDocs AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AskDocs AI")

st.write("Upload PDFs and ask questions about them.")

# Sidebar
with st.sidebar:

    st.header("Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        with st.spinner("Processing documents..."):

            for file in uploaded_files:

                with tempfile.NamedTemporaryFile(delete=False) as tmp:

                    tmp.write(file.read())
                    process_pdf(tmp.name)

        st.success("Documents processed!")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Knowledge Base"):

        reset_database()
        st.success("Vector database reset!")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, sources = ask_question(prompt)

            placeholder = st.empty()
            streamed = ""

            for char in answer:

                streamed += char
                placeholder.markdown(streamed)
                time.sleep(0.01)

            if sources:

                st.markdown("### Sources")

                for s in sources:
                    st.write("📄", s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

st.markdown("---")
st.caption("Built with LangChain + ChromaDB + OpenAI")
