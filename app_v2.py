import streamlit as st

from src.pdf_processor import extract_text_from_pdf
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import build_context
from src.gemini import GeminiClient


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi PDF RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background: #0F172A;
        border-right: 1px solid #1E293B;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .sidebar-logo {
        font-size: 25px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        color: #94A3B8;
        font-size: 13px;
        margin-bottom: 25px;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .hero-container {
        padding: 20px 0 30px 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        color: #A78BFA;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 46px;
        line-height: 1.1;
        font-weight: 750;
        color: #F8FAFC;
        margin: 0;
    }

    .hero-gradient {
        color: #A78BFA;
    }

    .hero-description {
        color: #94A3B8;
        font-size: 17px;
        margin-top: 12px;
        max-width: 720px;
        line-height: 1.6;
    }


    /* --------------------------------------------------------
       STATUS
    -------------------------------------------------------- */

    .status-card {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 12px;
        background: #111827;
        border: 1px solid #1E293B;
        margin-bottom: 20px;
    }

    .status-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.6);
    }

    .status-text {
        color: #CBD5E1;
        font-size: 13px;
    }


    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #1E293B;
        min-height: 100px;
    }

    .metric-icon {
        font-size: 22px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #F8FAFC;
    }

    .metric-label {
        color: #64748B;
        font-size: 12px;
        margin-top: 3px;
    }


    /* --------------------------------------------------------
       SECTION HEADERS
    -------------------------------------------------------- */

    .section-title {
        color: #F8FAFC;
        font-size: 20px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 13px;
        margin-bottom: 18px;
    }


    /* --------------------------------------------------------
       SOURCE CARDS
    -------------------------------------------------------- */

    .source-card {
        padding: 16px 18px;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #1E293B;
        margin: 8px 0;
    }

    .source-title {
        color: #E2E8F0;
        font-weight: 600;
        font-size: 14px;
    }

    .source-meta {
        color: #64748B;
        font-size: 12px;
        margin-top: 5px;
    }

    .source-score {
        color: #A78BFA;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       EMPTY STATE
    -------------------------------------------------------- */

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        border-radius: 18px;
        background: #0F172A;
        border: 1px dashed #334155;
        margin-top: 20px;
    }

    .empty-icon {
        font-size: 45px;
        margin-bottom: 15px;
    }

    .empty-title {
        color: #F8FAFC;
        font-size: 21px;
        font-weight: 650;
    }

    .empty-description {
        color: #64748B;
        font-size: 14px;
        margin-top: 8px;
    }


    /* --------------------------------------------------------
       CHAT
    -------------------------------------------------------- */

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        border: 1px solid #1E293B;
        margin-bottom: 12px;
    }

    div[data-testid="stChatInput"] {
        margin-top: 20px;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #334155;
    }

    .stButton > button:hover {
        border-color: #8B5CF6;
    }


    /* --------------------------------------------------------
       FILE UPLOADER
    -------------------------------------------------------- */

    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 1px dashed #475569;
        background: #111827;
    }


    /* --------------------------------------------------------
       DIVIDERS
    -------------------------------------------------------- */

    hr {
        border-color: #1E293B !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "pdfs": 0,
        "pages": 0,
        "chunks": 0,
    }


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def load_embedding_model():
    return EmbeddingModel()


@st.cache_resource
def load_gemini():
    return GeminiClient()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">📚 Multi PDF RAG</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Your intelligent document workspace'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents.",
    )

    st.divider()

    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    if uploaded_files:

        st.markdown("#### Uploaded")

        for file in uploaded_files:
            st.markdown(
                f"""
                <div style="
                    padding: 8px 10px;
                    margin: 4px 0;
                    border-radius: 8px;
                    background: #111827;
                    color: #CBD5E1;
                    font-size: 13px;
                ">
                    📄 {file.name}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        st.caption("No documents uploaded yet.")

    st.divider()

    # --------------------------------------------------------
    # PROCESS BUTTON
    # --------------------------------------------------------

    process_button = st.button(
        "⚡ Process Documents",
        use_container_width=True,
        type="primary",
        disabled=not uploaded_files,
    )

    if st.session_state.stats["pdfs"] > 0:

        st.divider()

        st.markdown("#### 📊 Workspace")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "PDFs",
                st.session_state.stats["pdfs"],
            )

        with col2:
            st.metric(
                "Pages",
                st.session_state.stats["pages"],
            )

        st.metric(
            "Indexed Chunks",
            st.session_state.stats["chunks"],
        )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_button:

    embedding_model = load_embedding_model()

    all_documents = []

    total_pages = 0

    progress = st.progress(
        0,
        text="Preparing documents...",
    )

    for file_index, uploaded_file in enumerate(uploaded_files):

        pages = extract_text_from_pdf(uploaded_file)

        total_pages += len(pages)

        for page in pages:

            all_documents.append(
                {
                    "text": page["text"],
                    "source": uploaded_file.name,
                    "page": page["page"],
                }
            )

        progress.progress(
            int(
                ((file_index + 1) / len(uploaded_files))
                * 100
            ),
            text=f"Processing {uploaded_file.name}...",
        )

    if not all_documents:

        st.error(
            "No readable text was found in the uploaded PDFs."
        )

    else:

        progress.progress(
            100,
            text="Creating embeddings...",
        )

        texts = [
            document["text"]
            for document in all_documents
        ]

        embeddings = embedding_model.embed_documents(texts)

        vector_store = VectorStore(
            dimension=embeddings.shape[1]
        )

        vector_store.add_documents(
            all_documents,
            embeddings,
        )

        st.session_state.vector_store = vector_store
        st.session_state.documents = all_documents

        st.session_state.stats = {
            "pdfs": len(uploaded_files),
            "pages": total_pages,
            "chunks": len(all_documents),
        }

        st.session_state.messages = []

        progress.empty()

        st.success(
            f"Successfully indexed {len(all_documents)} document sections."
        )

        st.rerun()


# ============================================================
# MAIN HERO
# ============================================================

st.markdown(
    """
  

            ✨ AI-Powered Document Intelligence

  

            Upload multiple PDFs and ask questions across your
            entire document collection using semantic search
            and Gemini-powered generation.

    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

if st.session_state.vector_store:

    st.markdown(
        """
        <div class="status-card">
            <div class="status-dot"></div>
            <div class="status-text">
                Knowledge base ready · Documents indexed successfully
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-card">
            <div class="status-dot"></div>
            <div class="status-text">
                Waiting for documents · Upload PDFs from the sidebar
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📄</div>
            <div class="metric-value">
                {st.session_state.stats["pdfs"]}
            </div>
            <div class="metric-label">
                Documents
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">📑</div>
            <div class="metric-value">
                {st.session_state.stats["pages"]}
            </div>
            <div class="metric-label">
                Pages indexed
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">🧠</div>
            <div class="metric-value">
                {st.session_state.stats["chunks"]}
            </div>
            <div class="metric-label">
                Searchable sections
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================


if not st.session_state.vector_store:

    st.markdown(
        """
<div class="empty-state">
<div class="empty-icon">📚</div>

<div class="empty-title">
    Your document workspace is empty</div>

<div class="empty-description">
        Upload one or more PDFs using the sidebar,
        then click <b>Process Documents</b> to build
        your searchable knowledge base.
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()



# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask your documents</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Questions are answered using the most relevant sections '
    'retrieved from your uploaded PDFs.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="🧑" if message["role"] == "user" else "✨",
    ):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                f"📚 Sources ({len(message['sources'])})"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <div class="source-title">
                        📄 {source["source"]}
                        </div>

                        <div class="source-meta">
                                Page {source["page"]}
                                &nbsp; · &nbsp;
                                Similarity:
                                <span class="source-score">
                                    {source["score"]:.3f}
                        </span>
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask anything about your documents..."
)


if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑",
    ):

        st.markdown(question)


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    embedding_model = load_embedding_model()

    query_embedding = embedding_model.embed_query(
        question
    )

    results = st.session_state.vector_store.search(
        query_embedding,
        top_k=5,
    )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="✨",
    ):

        with st.spinner(
            "Searching your documents..."
        ):

            context = build_context(results)

            gemini = load_gemini()

            answer = gemini.generate_answer(
                question,
                context,
            )

        st.markdown(answer)

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if results:

            with st.expander(
                f"📚 Sources ({len(results)})"
            ):

                for result in results:

                    st.markdown(
                        f"""
                        <div class="source-card">

                        <div class="source-title">
                                📄 {result["source"]}
                        </div>

                        <div class="source-meta">
                                Page {result["page"]}
                                &nbsp; · &nbsp;
                                Similarity:
                                <span class="source-score">
                                    {result["score"]:.3f}
                            </span>
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )