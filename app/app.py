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
from typing import List
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

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
            model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    examples = [
                {
                    "pergunta": "Como posso utilizar morfina para dor?",
                    "resposta": """
            Pergunta de acompanhamento necessária: Sim.
            1. Pergunta de acompanhamento: Existe referência a algum capítulo no texto?
            Resposta intermediária: Sim.
            2. Pergunta de acompanhamento: Qual capítulo é relevante?
            Resposta intermediária: Dor.
            3. Pergunta de acompanhamento: Dentro do capítulo 'Dor', existe referência a algum subtítulo?
            Resposta intermediária: Sim.
            4. Pergunta de acompanhamento: Qual subtítulo?
            Resposta intermediária: Morfina.

            Resposta final:
            Segundo o Manual de Cuidados Paliativos, 2ª ed.:
            Para dor, a morfina pode ser utilizada da seguinte forma:
            Morfina simples:
            - Dose inicial: 5 mg a cada 4 horas (VO), com necessidade de avaliar doses mais baixas em pacientes idosos, com disfunção renal ou hepática;
            - Dose máxima: Não possui dose teto; o limite é o efeito colateral, devendo ser titulado cuidadosamente;
            - Frequência de administração: A cada 4 horas. Em casos específicos (idosos, disfunções), considerar a cada 6 horas;
            - Vias de administração: Oral, sonda nasoenteral, gastrostomia, endovenosa, subcutânea, hipodermóclise;
            - Equipotência: Morfina endovenosa é três vezes mais potente que a oral;
            - Particularidades: Metabolizada no fígado e excretada pelo rim. Usar com cautela em pacientes com doença hepática ou renal;
            - Disponibilidade no SUS: Constante na Rename 2022.
            """,
                },
                {
                    "pergunta": "Quais são os efeitos colaterais da morfina?",
                    "resposta": """
            Pergunta de acompanhamento necessária: Sim.
            1. Existe um capítulo relacionado? Sim.
            Resposta intermediária: Dor.
            2. Algum subtítulo é relevante? Sim.
            Resposta intermediária: Efeitos colaterais.

            Resposta final:
            Segundo o Manual de Cuidados Paliativos, 2ª ed.:
            Os principais efeitos colaterais da morfina incluem náuseas, vômitos, constipação, sonolência e depressão respiratória. Esses efeitos devem ser monitorados e manejados adequadamente.
            """
                },
                {
                    "pergunta": "Quem são os autores do Capitulo de Dor?",
                    "resposta": """
            Pergunta de acompanhamento necessária: Sim.
            1. Existe um capítulo relacionado? Sim.
            Resposta intermediária: Dor.
            2. Algum subtítulo é relevante? Não.
            Resposta intermediária: Efeitos colaterais.
            3. Foi perguntado sobre nomes de pessoas? Sim.
            Resposta intermediária: Sim.
            4. É sobre o livro ou capítulo?
            Resposta intermediária: capítulo.


            Resposta final:
            Os autores do Capítulo de Dor do Manual de Cuidados Paliativos, 2ª ed., são:
            1- Daniel Felgueiras Rolo
            2- Maria Perez Soares D'Alessandro
            3- Gustavo Cassefo
            4- Sergio Seiki Anagusko
            5- Ana Paula Mirarchi Vieira Maiello
            """
                },
                {
                    "pergunta": "Quem são os autores do livro?",
                    "resposta": """
            Pergunta de acompanhamento necessária: Sim.
            1. Existe um capítulo relacionado? 
            Resposta intermediária: Não.
            2. Algum subtítulo é relevante? 
            Resposta intermediária: Não.
            3. Foi perguntado sobre nomes de pessoas? Sim.
            Resposta intermediária: Sim.
            4. É sobre o livro ou capítulo?
            Resposta intermediária: livro.


            Resposta final:
            A equipe responsável pelo Manual de Cuidados Paliativos, 2ª ed., foram:

            Editores
            Maria Perez Soares D’Alessandro, Lara Cruvinel Barbosa, Sergio Seiki Anagusko, Ana Paula Mirarchi Vieira
            Maiello, Catherine Moreira Conrado, Carina Tischler Pires e Daniel Neves Forte.

            Autores
            Aline de Almada Messias, Ana Cristina Pugliese de Castro, Ana Paula Mirarchi Vieira Maiello, Caroline
            Freitas de Oliveira, Catherine Moreira Conrado, Daniel Felgueiras Rolo, Fábio Holanda Lacerda, Fernanda
            Pimentel Coelho, Fernanda Spiel Tuoto, Graziela de Araújo Costa , Gustavo Cassefo, Heloisa Maragno,
            Hieda Ludugério de Souza, Lara Cruvinel Barbosa, Leonardo Bohner Hoffmann, Lícia Maria Costa Lima,
            Manuele de Alencar Amorim, Marcelo Oliveira Silva, Maria Perez Soares D’Alessandro, Mariana Aguiar
            Bezerra, Nathalia Maria Salione da Silva, Priscila Caccer Tomazelli, Sergio Seiki Anagusko e Sirlei Dal Moro.

            Equipe da Secretaria de Atenção Especializada em Saúde do Ministério da Saúde
            Nilton Pereira Junior, Mariana Borges Dias, Taís Milene Santos de Paiva, Cristiane Maria Reis Cristalda e
            Lorayne Andrade Batista.

            Equipe do CONASS
            René José Moreira dos Santos, Eliana Maria Ribeiro Dourado e Luciana Toledo.

            Equipe de apoio HSL:
            Guilherme Fragoso de Mello, Juliana de Lima Gerarduzzi e Luiz Felipe Monteiro Correia.
            """
                }
            ]
    try:
        groq_key, _ = load_api_keys()
        chat_model = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.2-3b-preview",
            temperature=0.3,
            max_tokens=1028
        )

        # Template para os exemplos
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Pergunta: {pergunta}\nContexto: {context}\nResposta:")
        ])

        # Integração dos exemplos
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # Template final com system message e exemplos
        final_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
Você é um Chatbot que auxilia profissionais de saúde em cuidados paliativos com base apenas no Manual de Cuidados Paliativos, 2ª ed., São Paulo: Hospital Sírio-Libanês; Ministério da Saúde, 2023.
Responda apenas com informações documentadas no manual e, caso não saiba a resposta, pergunte se pode buscar em outras fontes.
Estruture as respostas de forma clara, mencionando capítulos e subtítulos do manual quando relevante.
"""
            ),
            few_shot_prompt,
            ("human", question)
        ])

        # Preparar o contexto
        context_text = " ".join(doc.page_content for doc in context)
        
        # Gerar as mensagens formatadas
        messages = final_prompt.format_messages(
            pergunta=question,
            context=context_text
        )
        
        # Obter a resposta do modelo
        response = chat_model.invoke(messages)
        return response.content
    
    except Exception as e:
        return f"Erro ao processar a pergunta: {str(e)}"

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
