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
import requests
import warnings

# Configurações para suprimir avisos
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuração de cache para as chaves API
@st.cache_data
def load_api_keys() -> Tuple[str, str, str, str]:
    """Carrega as chaves API e as chaves de descriptografia do Mega.nz."""
    try:
        return (
            st.secrets["api_keys"]["groq_api_key"],
            st.secrets["api_keys"]["hf_api_key"],
            st.secrets["api_keys"]["store_key_json"],
            st.secrets["api_keys"]["index_key_pkl"]
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar chaves API. Verifique as configurações.")
        raise ValueError(f"Erro nas chaves API: {e}")

def download_file(url: str, decrypt_key: str, output_path: str):
    """
    Faz o download de um arquivo dado um link público e uma chave de descriptografia.

    Args:
        url (str): URL base do Mega.nz.
        decrypt_key (str): Chave de descriptografia do arquivo.
        output_path (str): Caminho onde o arquivo será salvo.
    """
    try:
        # Constrói o link completo
        full_url = f"{url}#{decrypt_key}"
        response = requests.get(full_url, stream=True)
        response.raise_for_status()  # Lança erro para códigos de resposta HTTP >= 400
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Arquivo baixado e salvo em: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar o arquivo de {url}: {e}")

# Carrega as chaves
groq_api_key, hf_api_key, store_key_json, index_key_pkl = load_api_keys()

# URLs base dos arquivos no Mega.nz
store_data_url = "https://mega.nz/file/3FUhULwC"
index_pkl_url = "https://mega.nz/file/SYNmXYjK"

store_data_path = "data/store_data.json"
index_pkl_path = "data/index.pkl"

# Faz download dos arquivos, incluindo as chaves de descriptografia
if not os.path.exists(store_data_path):
    download_file(store_data_url, store_key_json, store_data_path)
if not os.path.exists(index_pkl_path):
    download_file(index_pkl_url, index_key_pkl, index_pkl_path)



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
            model_name="neuralmind/bert-base-portuguese-cased"
        )
    except Exception as e:
        st.error("⚠️ Erro na inicialização dos embeddings")
        raise ValueError(f"Erro nos embeddings: {e}")

@st.cache_resource
def initialize_vector_store() -> FAISS:
    """Inicializa e carrega o índice FAISS."""
    # Carrega as chaves
    groq_api_key, hf_api_key, store_key_json, index_key_pkl = load_api_keys()
    
    # URLs base dos arquivos no Mega.nz
    store_data_url = "https://mega.nz/file/3FUhULwC"
    index_pkl_url = "https://mega.nz/file/SYNmXYjK"
    
    store_data_path = "data/store_data.json"
    index_pkl_path = "data/index.pkl"
    
    # Faz download dos arquivos, incluindo as chaves de descriptografia
    if not os.path.exists(store_data_path):
        download_file(store_data_url, store_key_json, store_data_path)
    if not os.path.exists(index_pkl_path):
        download_file(index_pkl_url, index_key_pkl, index_pkl_path)
    try:
        embeddings = initialize_embeddings()
        index_path = Path(__file__).parent / "faiss_index"
        
        # Para debug
        #st.write(f"Tentando carregar de: {index_path}")
        #st.write(f"O diretório existe? {index_path.exists()}")
        
        if not index_path.exists():
            raise FileNotFoundError(f"📁 Diretório do índice FAISS não encontrado em {index_path}")
            
        return FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar índice FAISS")
        st.write(f"Diretório atual: {Path.cwd()}")  # Mostra diretório atual
        raise ValueError(f"Erro no FAISS: {e}")

def get_chat_response(context: List[Document], question: str) -> str:
    """Processa a pergunta e retorna a resposta do modelo."""
    try:
        groq_key, _ = load_api_keys()
        chat_model = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.2-3b-preview",
            temperature=0.3,
            max_tokens=1028
        )

        system_prompt = """Você é um Chatbot especializado em cuidados paliativos, baseando-se exclusivamente no Manual de Cuidados Paliativos, 2ª ed., São Paulo: Hospital Sírio-Libanês; Ministério da Saúde, 2023.
        - Responda apenas com informações documentadas no manual
        - Forneça orientações detalhadas sobre medicações
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
    # `st.set_page_config` deve ser o primeiro comando Streamlit
    st.set_page_config(
        page_title="Chatbot - Manual de Cuidados Paliativos",
        page_icon="🏥",
        layout="wide"
    )
    
    # Restante do código da função
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
