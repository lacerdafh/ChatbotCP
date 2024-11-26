<<<<<<< HEAD
# Chatbot do Manual de Cuidados Paliativos - 2ª edição

Um chatbot especializado desenvolvido para facilitar o acesso e consulta ao Manual de Cuidados Paliativos (2ª edição revisada e ampliada). Utilizando tecnologias avançadas como FAISS (Facebook AI Similarity Search) para busca semântica e Groq LLM para processamento de linguagem natural, o sistema permite consultas interativas e respostas contextualizadas sobre o conteúdo do manual.

## 📋 Sobre o Projeto

Este chatbot foi desenvolvido com o objetivo de democratizar o acesso às informações contidas no Manual de Cuidados Paliativos, permitindo que profissionais de saúde e interessados possam consultar rapidamente informações específicas através de perguntas em linguagem natural. O sistema utiliza:

- FAISS para busca semântica eficiente
- Groq LLM para processamento de linguagem natural
- HuggingFace Embeddings para vetorização de texto
- Interface interativa construída com Streamlit

## 📋 Sobre o livro

Disponivel em https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/guias-e-manuais/2023/manual-de-cuidados-paliativos-2a-edicao/view

- Editores
Maria Perez Soares D’Alessandro, Lara Cruvinel Barbosa, Sergio Seiki Anagusko, Ana Paula Mirarchi Vieira
Maiello, Catherine Moreira Conrado, Carina Tischler Pires e Daniel Neves Forte.

- Autores
Aline de Almada Messias, Ana Cristina Pugliese de Castro, Ana Paula Mirarchi Vieira Maiello, Caroline
Freitas de Oliveira, Catherine Moreira Conrado, Daniel Felgueiras Rolo, Fábio Holanda Lacerda, Fernanda
Pimentel Coelho, Fernanda Spiel Tuoto, Graziela de Araújo Costa , Gustavo Cassefo, Heloisa Maragno,
Hieda Ludugério de Souza, Lara Cruvinel Barbosa, Leonardo Bohner Hoffmann, Lícia Maria Costa Lima,
Manuele de Alencar Amorim, Marcelo Oliveira Silva, Maria Perez Soares D’Alessandro, Mariana Aguiar
Bezerra, Nathalia Maria Salione da Silva, Priscila Caccer Tomazelli, Sergio Seiki Anagusko e Sirlei Dal Moro.

- Equipe da Secretaria de Atenção Especializada em Saúde do Ministério da Saúde
Nilton Pereira Junior, Mariana Borges Dias, Taís Milene Santos de Paiva, Cristiane Maria Reis Cristalda e
Lorayne Andrade Batista.

- Equipe do CONASS
René José Moreira dos Santos, Eliana Maria Ribeiro Dourado e Luciana Toledo.

- Equipe de apoio HSL:
Guilherme Fragoso de Mello, Juliana de Lima Gerarduzzi e Luiz Felipe Monteiro Correia.

## 🚀 Estrutura do Projeto

```
app/
├── static/
│   └── images/
│       └── app_header.png
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── config.toml
├── requirements.txt
└── app.py
```

## 🔧 Requisitos

- Python 3.9+
- Groq API Key
- HuggingFace API Key

## 📦 Dependências Principais

- streamlit
- langchain
- langchain-groq
- python-dotenv
- faiss-cpu
- sentence-transformers
- huggingface-hub
- pydantic

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/lacerdafh/ChatbotCP.git
cd chatbot-dr-kinho
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as chaves de API:
   - Crie um arquivo `config.toml` na pasta raiz do projeto
   - Adicione suas chaves de API:
```toml
[api_keys]
groq_api_key = "sua-groq-api-key"
hf_api_key = "sua-huggingface-api-key"
```

## 🚀 Como Executar

1. Navegue até a pasta do projeto
2. Execute o comando:
```bash
streamlit run app/app.py
```

## 💻 Como Usar

1. Após iniciar a aplicação, acesse através do navegador (geralmente em `http://localhost:8501`)
2. Digite sua pergunta no campo de texto
3. O sistema irá:
   - Buscar informações relevantes na base de conhecimento
   - Processar a pergunta usando o modelo de linguagem
   - Retornar uma resposta contextualizada
   - Mostrar as fontes consultadas

## 🔍 Características

- Interface intuitiva e responsiva
- Respostas baseadas em documentos pré-processados
- Busca semântica de alta performance
- Citação de fontes consultadas
- Sistema de cache para respostas rápidas

## ⚠️ Notas Importantes

- O índice FAISS é fixo e pré-construído
- As respostas são baseadas apenas nos documentos incluídos no índice
- É necessário ter as chaves de API configuradas corretamente

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

## 📄 Licença

Este projeto está sob a licença Apache LicenseVersion 2.0, January 2004.  Veja o arquivo LICENSE para mais detalhes.

## 🎯 Contato

Para dúvidas ou sugestões, por favor abra uma issue no repositório.

---
Desenvolvido por Fábio Lacerda- (aka. lacerdafh @_fabio_lacerda) para auxiliar no acesso à informação sobre Cuidados Paliativos
=======
# ChatbotCP
Um chatbot especializado baseado em documentos utilizando FAISS (Facebook AI Similarity Search) para busca semântica e Groq LLM para processamento de linguagem natural.

