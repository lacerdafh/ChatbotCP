import os
from typing import Tuple, List
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pathlib import Path
import warnings

# Configurações para suprimir avisos
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Configuração de cache para as chaves API
@st.cache_data
def load_api_keys() -> Tuple[str, str]:
    """Carrega as chaves API necessárias."""
    try:
        return (
            st.secrets["api_keys"]["hf_api_key"],
            st.secrets["api_keys"]["store_key_json"]
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar chaves API. Verifique as configurações.")
        raise ValueError(f"Erro nas chaves API: {e}")

@st.cache_resource
def initialize_embeddings() -> HuggingFaceInferenceAPIEmbeddings:
    """Inicializa o modelo de embeddings com cache."""
    try:
        hf_key, _ = load_api_keys()
        os.environ["HF_API_KEY"] = hf_key
        
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_key,
            model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    except Exception as e:
        st.error("⚠️ Erro na inicialização dos embeddings")
        raise ValueError(f"Erro nos embeddings: {e}")

@st.cache_resource
def initialize_vector_store() -> FAISS:
    """Inicializa e carrega o índice FAISS."""
    try:
        embeddings = initialize_embeddings()
        index_path = Path(__file__).parent / "faiss_index"

        # Verifica se o índice FAISS existe
        if not index_path.exists():
            raise FileNotFoundError(f"📁 Diretório do índice FAISS não encontrado em {index_path}")

        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar índice FAISS")
        raise ValueError(f"Erro no FAISS: {e}")

def render_sidebar():
    """Renderiza a sidebar do aplicativo."""
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.markdown("""
        💬 **Assistente baseado em embeddings do modelo BiomedCLIP**
        
        📚 **Dicas de uso:**
        - Faça perguntas claras e específicas
        - Utilize termos técnicos médicos
        - Forneça contexto clínico relevante
        """)

def main():
    st.set_page_config(
        page_title="Chatbot - Manual de Cuidados Paliativos",
        page_icon="🏥",
        layout="wide"
    )
    st.title("🤖 Chatbot do Manual de Cuidados Paliativos")
    render_sidebar()

    try:
        # Inicialização do vector store
        if 'vector_store' not in st.session_state:
            with st.spinner("📚 Carregando base de conhecimento..."):
                st.session_state.vector_store = initialize_vector_store()
                st.success("✅ Base de conhecimento carregada!")

        # Configuração do retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )

        # Interface do usuário
        user_question = st.text_input(
            "💭 Faça sua pergunta sobre o Manual de Cuidados Paliativos:",
            key="user_input"
        )
        
        if user_question:
            with st.spinner("🔄 Processando sua pergunta..."):
                context = retriever.get_relevant_documents(user_question)
                response = "🔄 Resposta simulada: Integração com BiomedCLIP ainda em progresso."

                # Exibição da resposta
                st.markdown("### 📝 Resposta:")
                st.markdown(response)

                # Exibição das fontes
                with st.expander("📚 Fontes consultadas"):
                    sources = set(doc.metadata.get('source', 'Desconhecido') 
                                for doc in context)
                    for source in sources:
                        st.markdown(f"- {Path(source).name}")

    except Exception as e:
        st.error(f"⚠️ Erro no aplicativo: {str(e)}")
        st.info("🔄 Tente recarregar a página ou contate o suporte.")

if __name__ == "__main__":
    main()
