import os
from typing import Tuple, List
import streamlit as st
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain.schema import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import warnings
import faiss
import json

# Suprimir avisos
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar configurações e chaves API
def load_api_keys() -> Tuple[str, str]:
    """Carrega as chaves API necessárias."""
    try:
        return (
            st.secrets["api_keys"]["groq_api_key"],
            st.secrets["api_keys"]["hf_api_key"]
        )
    except Exception as e:
        st.error("⚠️ Erro ao carregar chaves API. Verifique as configurações.")
        raise ValueError(f"Erro nas chaves API: {e}")

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings específico do DPR."""
    try:
        groq_api_key, hf_api_key = load_api_keys()
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["HF_API_KEY"] = hf_api_key
        
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_api_key,
            model_name="facebook/dpr-ctx_encoder-multiset-base"
        )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings: {str(e)}")
        raise

def load_faiss_index(index_path: str, embeddings):
    """Carrega o índice FAISS específico."""
    try:
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            raise FileNotFoundError(f"Arquivo index.faiss não encontrado em: {index_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar index.faiss: {str(e)}")

def main():
    st.title("Chatbot com Base de Conhecimento FAISS")

    # Configurar caminho do index.faiss
    current_dir = Path(os.path.dirname(__file__))  # Diretório do arquivo atual
    index_path = current_dir / "index.faiss"
    # Inicializar embeddings e carregar índice FAISS
    try:
        embeddings = get_embeddings()
        vector_store = load_faiss_index(index_path, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        st.success("Base de conhecimento carregada com sucesso!")
    except Exception as e:
        st.error(f"Erro ao inicializar o sistema: {str(e)}")
        return

    # Interface do chat
    st.write("### Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Exibir mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Campo de entrada do usuário
    user_question = st.chat_input("Digite sua pergunta:")

    if user_question:
        # Adicionar pergunta do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        try:
            with st.spinner("Processando sua pergunta..."):
                # Recuperar contexto relevante do FAISS
                context = retriever.get_relevant_documents(user_question)
                
                # Configurar chat model
                chat_model = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name="llama-3.2-3b-preview",
                    temperature=0.4,
                    max_tokens=512
                )

                # Preparar prompt com contexto
                messages = [
                    ("system", """Você é um assistente especializado que responde apenas com base no 
                     contexto fornecido. Se a informação não estiver no contexto, diga que não pode 
                     responder. Seja conciso e direto em suas respostas."""),
                    ("user", f"""
                    Contexto: {' '.join(doc.page_content for doc in context)}
                    
                    Pergunta: {user_question}
                    """)
                ]

                # Gerar resposta
                response = chat_model.invoke(messages)

                # Exibir resposta
                with st.chat_message("assistant"):
                    st.write(response.content)
                
                # Adicionar resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": response.content})

                # Exibir trechos relevantes em um expander
                with st.expander("Ver trechos relevantes utilizados"):
                    for i, doc in enumerate(context, 1):
                        st.markdown(f"**Trecho {i}:**")
                        st.write(doc.page_content[:200] + "...")

        except Exception as e:
            st.error(f"Erro ao processar pergunta: {str(e)}")

if __name__ == "__main__":
    main()
