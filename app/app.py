import os
import streamlit as st
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS

# Suprimir avisos
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["GROQ_API_KEY"] = st.secrets["api_keys"]["groq_api_key"]
os.environ["HF_API_KEY"] = st.secrets["api_keys"]["hf_api_key"]

def get_api_keys():
    try:
        groq_api_key = st.secrets["api_keys"]["groq_api_key"]
        hf_api_key = st.secrets["api_keys"]["hf_api_key"]
        return groq_api_key, hf_api_key
    except Exception as e:
        st.error("Erro ao carregar chaves de API. Verifique as configurações de secrets.")
        raise e

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings."""
    try:
        hf_api_key = os.getenv("HF_API_KEY")
        if not hf_api_key:
            raise ValueError("HF_API_KEY não encontrada")

        return HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings: {str(e)}")
        raise

@st.cache_resource
def load_vector_store():
    """Carrega o índice FAISS existente."""
    try:
        embeddings = get_embeddings()
        index_path = Path("faiss_index")
        
        if not index_path.exists():
            raise FileNotFoundError("Diretório do índice FAISS não encontrado")
            
        vector_store = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        st.error(f"Erro ao carregar índice FAISS: {str(e)}")
        raise

def get_chat_response(context, user_question):
    try:
        chat_model = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.2-3b-preview",
            temperature=0.4,
            max_tokens=1028
        )

        system_message = SystemMessage(
            content='''Você é um Chatbot que auxilia profissionais de saúde em cuidados paliativos com base apenas no Manual de Cuidados Paliativos, 2ª ed., São Paulo: Hospital Sírio-Libanês; Ministério da Saúde, 2023.
                    Responda apenas com informações documentadas no manual e, deve orientar sobre todo tipo de medicação de forma completa!
                    Estruture as respostas de forma clara, mencionando capítulos e subtítulos do manual quando relevante.'''
        )

        user_content = f"""
        Contexto: {' '.join(doc.page_content for doc in context)}
        Pergunta: {user_question}
        """
        human_message = HumanMessage(content=user_content)

        messages = [system_message, human_message]
        response = chat_model.invoke(messages)
        
        return response.content
    except Exception as e:
        st.error(f"Erro ao processar resposta: {str(e)}")
        raise

def main():
    st.title("Chatbot do Manual de Cuidados Paliativos")

    # Sidebar
    with st.sidebar:
        image_path = Path("static/images/app_header.png")
        if image_path.exists():
            st.image(str(image_path), caption="Familia CP-Sirio tentando levar conhecimento a todos", use_container_width=True)
        
        st.header("Informações")
        st.write("💬 Assistente baseado no Manual de Cuidados Paliativos 2ª Ed.")
        st.write("📚 Use perguntas claras e específicas")

    # Carregar vector store
    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Carregando base de conhecimento..."):
                st.session_state.vector_store = load_vector_store()
                st.success("Base de conhecimento carregada!")

        # Configurar retriever
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

        # Interface de chat
        user_question = st.text_input("Faça sua pergunta sobre o Manual de Cuidados Paliativos:")
        
        if user_question:
            with st.spinner("Processando..."):
                # Recuperar documentos relevantes
                context = retriever.get_relevant_documents(user_question)

                # Obter resposta
                response = get_chat_response(context, user_question)

                # Exibir resposta
                with st.container():
                    st.markdown("### Resposta:")
                    st.write(response)

                    st.markdown("### Fontes consultadas:")
                    sources = set(doc.metadata.get('source', 'Desconhecido') for doc in context)
                    for source in sources:
                        st.write(f"- {os.path.basename(source)}")

    except Exception as e:
        st.error(f"Erro no aplicativo: {str(e)}")

if __name__ == "__main__":
    main()
