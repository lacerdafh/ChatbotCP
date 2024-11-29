import os
from typing import Tuple, List
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
import shutil
from nova_vectorestore import VectorStoreFlatMMR

# Configurações para suprimir avisos
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuração de cache para as chaves API
@st.cache_data
def load_api_keys() -> Tuple[str, str]:
    """Carrega as chaves API do Streamlit Secrets."""
    try:
        return (
            st.secrets["api_keys"]["groq_api_key"],
            st.secrets["api_keys"]["hf_api_key"]
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar chaves API. Verifique as configurações.")
        raise ValueError(f"Erro nas chaves API: {e}")

# Inicialização do modelo de embeddings
@st.cache_resource
def initialize_embeddings() -> HuggingFaceInferenceAPIEmbeddings:
    """Inicializa o modelo de embeddings com cache."""
    try:
        groq_key, hf_key = load_api_keys()
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["HF_API_KEY"] = hf_key
        
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_key,
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception as e:
        st.error("⚠️ Erro na inicialização dos embeddings")
        raise ValueError(f"Erro nos embeddings: {e}")

@st.cache_resource
def initialize_vector_store() -> VectorStoreFlatMMR:
    """Inicializa e carrega o índice com a classe VectorStoreFlatMMR."""
    try:
        embeddings = initialize_embeddings()
        index_path = Path("app\\faiss_index")  # Caminho para o novo índice

        # Verificar se o arquivo do índice existe
        if not index_path.exists():
            raise FileNotFoundError(f"📁 Arquivo do índice FAISS não encontrado em {index_path}")

        # Instanciar a classe VectorStoreFlatMMR
        vector_store = VectorStoreFlatMMR(
            embedding_model="neuralmind/bert-base-portuguese-cased",  # Modelo escolhido
            lambda_param=0.7,  # Ajustável para balancear relevância e diversidade
            top_k=10,  # Número de documentos retornados por busca
            max_vectors_warning=100000,  # Aviso ao ultrapassar limite de vetores
            chunk_size=1000,  # Tamanho dos chunks de texto
            chunk_overlap=200  # Sobreposição entre chunks
        )

        # Carregar o índice FAISS
        vector_store.index = vector_store.load_vector_store(str(index_path))

        return vector_store
    except Exception as e:
        st.error("⚠️ Erro ao carregar índice FAISS com VectorStoreFlatMMR")
        raise ValueError(f"Erro no FAISS: {e}")

def get_chat_response(context: List[Document], question: str) -> str:
    """Processa a pergunta e retorna a resposta do modelo."""
    try:
        groq_key, _ = load_api_keys()
        chat_model = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.2-3b-preview",
            temperature=0.5,
            max_tokens=1028
        )

        system_prompt = """Você é um Chatbot especializado em cuidados paliativos, baseando-se exclusivamente no Manual de Cuidados Paliativos, 2ª ed., São Paulo: Hospital Sírio-Libanês; Ministério da Saúde, 2023.
        - Responda apenas com informações documentadas no manual
        - obrigatorio fornecer orientações detalhadas sobre medicações
        - Estruture as respostas de forma clara
        - Mencione capítulos e subtítulos relevantes do manual"""

        context_text = " ".join(doc.page_content for doc in context)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Contexto: {context_text}\nPergunta: {question}")
        ]
        
        return chat_model.invoke(messages).content
    except Exception as e:
        st.error("⚠️ Erro ao processar resposta")
        return f"Desculpe, ocorreu um erro: {str(e)}"

def render_sidebar():
    """Renderiza a sidebar do aplicativo."""
    with st.sidebar:
        image_path = Path(__file__).parent / "static" / "images" / "app_header.png"
        if image_path.exists():
            st.image(
                str(image_path),
                caption="Familia CP-Sirio tentando levar conhecimento a todos",
                use_container_width=True
            )
        
        st.header("ℹ️ Informações")
        st.markdown("""
        💬 **Assistente baseado no Manual de Cuidados Paliativos 2ª Ed.**
        
        📚 **Dicas de uso:**
        - Use perguntas claras e específicas
        - Mencione termos técnicos corretamente
        - Indique o contexto clínico quando relevante
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
            search_kwargs={"k": 10}
        )

        # Interface do usuário
        user_question = st.text_input(
            "💭 Faça sua pergunta sobre o Manual de Cuidados Paliativos:",
            key="user_input"
        )
        
        if user_question:
            with st.spinner("🔄 Processando sua pergunta..."):
                context = retriever.get_relevant_documents(user_question)
                response = get_chat_response(context, user_question)

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
