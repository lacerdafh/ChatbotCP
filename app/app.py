import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Chatbot - Manual de Cuidados Paliativos",
    page_icon="💬",
    layout="centered"
)

# Título principal
st.title("💬 Chatbot do Manual de Cuidados Paliativos")

# Corpo da mensagem
st.markdown(
    """
    ## Estamos melhorando nosso chatbot!
    Em breve, teremos novidades incríveis para tornar o Manual de Cuidados Paliativos ainda mais acessível e útil.

    Fique de olho! 👀
    """
)

# Imagem ou animação (opcional)
st.image(
    "https://via.placeholder.com/800x400.png?text=Novidades+em+Breve!",
    use_column_width=True
)

# Rodapé
st.markdown("---")
st.markdown("© 2024 - Equipe de Cuidados Paliativos")
