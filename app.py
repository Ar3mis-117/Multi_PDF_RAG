import streamlit as st

from src.gemini import GeminiClient
from src.rag import build_context
from src.pdf_processor import extract_text_from_pdf
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


@st.cache_resource
def load_gemini():
    return GeminiClient()

st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Multi-PDF RAG Assistant")
st.write("Upload multiple PDFs and ask questions about their contents.")


@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


embedding_model = load_embedding_model()


uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)


if uploaded_files:

    st.subheader("📄 Uploaded Documents")

    for file in uploaded_files:
        st.write(f"• {file.name}")

    if st.button("🔨 Process Documents"):

        with st.spinner("Processing documents..."):

            all_chunks = []

            for uploaded_file in uploaded_files:

                pages = extract_text_from_pdf(uploaded_file)

                for page in pages:

                    all_chunks.append(
                        {
                            "text": page["text"],
                            "source": uploaded_file.name,
                            "page": page["page"],
                        }
                    )

            if not all_chunks:
                st.error("No readable text was found in the uploaded PDFs.")
                st.stop()

            texts = [
                chunk["text"]
                for chunk in all_chunks
            ]

            embeddings = embedding_model.embed_documents(texts)

            vector_store = VectorStore(
                dimension=embeddings.shape[1]
            )

            vector_store.add_documents(
                all_chunks,
                embeddings,
            )

            st.session_state.vector_store = vector_store

        st.success(
            f"Successfully processed {len(uploaded_files)} PDF(s) "
            f"with {len(all_chunks)} pages."
        )


if "vector_store" in st.session_state:

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "What would you like to know about your documents?"
    )

    if question:

        with st.spinner("Searching your documents..."):

            query_embedding = embedding_model.embed_query(
                question
            )

            results = st.session_state.vector_store.search(
                query_embedding,
                top_k=5,
            )

        context = build_context(results)

        with st.spinner("Generating answer..."):

            gemini = load_gemini()

            answer = gemini.generate_answer(
                question,
                context,
            )

        st.subheader("🤖 Answer")

        st.write(answer)

        st.subheader("📚 Sources")

        for result in results:

            st.write(
                f"📄 **{result['source']}** — "
                f"Page {result['page']} "
                f"(similarity: {result['score']:.3f})"
            )