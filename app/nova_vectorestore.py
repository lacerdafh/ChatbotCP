import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline
import json
from tqdm import tqdm
import os
import re
import logging
from datetime import datetime


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_texts_from_json(filepath: str) -> tuple[List[str], List[Dict]]:
        """
        Carrega textos e metadados do arquivo JSON
        
        Returns:
            texts: Lista de textos dos chunks
            metadatas: Lista de metadados correspondentes
        """
        texts = []
        metadatas = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chapters_data = json.load(f)
            
            for chapter_num, chapter_data in chapters_data.items():
                chapter_title = chapter_data['chapter_info']['title']
                
                if len(chapter_data['content']['textos']) > 0:
                    chapter_text = chapter_data['content']['textos'][0]['text']
                    
                    texts.append(chapter_text)
                    metadatas.append({
                        'chapter': chapter_title,
                        'chapter_number': chapter_num,
                    })
                    
            logger.info(f"Carregados {len(texts)} textos de {len(chapters_data)} capítulos")
            return texts, metadatas
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo JSON: {e}")
            raise

@dataclass
class Document:
    """Classe para representar um documento na vector store"""
    page_content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validação básica do documento"""
        if not self.page_content or not isinstance(self.page_content, str):
            raise ValueError("page_content deve ser uma string não vazia")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata deve ser um dicionário")

class VectorStoreFlatMMR:
    def __init__(
        self,
        embedding_model: str = "neuralmind/bert-base-portuguese-cased",  # Modelo em português
        lambda_param: float = 0.7,
        top_k: int = 5,
        max_vectors_warning: int = 100000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Vector Store otimizada com IndexFlatIP e MMR
        
        Args:
            embedding_model: Modelo para gerar embeddings (otimizado para português)
            lambda_param: Parâmetro MMR (0.7 para balancear relevância e diversidade)
            top_k: Número de resultados a retornar
            max_vectors_warning: Limite para alertar sobre tamanho do índice
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
        """
        self.embeddings = SentenceTransformer(embedding_model)
        self.lambda_param = lambda_param
        self.top_k = top_k
        self.max_vectors_warning = max_vectors_warning
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.dimension = self.embeddings.get_sentence_embedding_dimension()
        self.documents: List[Document] = []
        
        # Inicializa índice Flat
        self.index = faiss.IndexFlatIP(self.dimension)

    def get_index_info(self) -> str:
        """Retorna informações detalhadas sobre o índice"""
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "memory_usage_mb": (self.index.ntotal * self.dimension * 4) / (1024 * 1024)
        }
        
        info = f"""
        📊 Informações do Índice FAISS
        
        Configuração:
        - Tipo: IndexFlatIP
        - Dimensão: {stats['dimension']}
        - Lambda MMR: {self.lambda_param}
        - Top K: {self.top_k}
        - Chunk Size: {self.chunk_size}
        - Chunk Overlap: {self.chunk_overlap}
        
        Status Atual:
        - Vetores: {stats['total_vectors']:,} / {self.max_vectors_warning:,} (máx. recomendado)
        - Documentos: {stats['total_documents']:,}
        - Memória: {stats['memory_usage_mb']:.2f} MB
        """
        return info  

    def load_texts_from_json(filepath: str) -> tuple[List[str], List[Dict]]:
        """
        Carrega textos e metadados do arquivo JSON
        
        Returns:
            texts: Lista de textos dos chunks
            metadatas: Lista de metadados correspondentes
        """
        texts = []
        metadatas = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chapters_data = json.load(f)
            
            for chapter_num, chapter_data in chapters_data.items():
                chapter_title = chapter_data['chapter_info']['title']
                
                if len(chapter_data['content']['textos']) > 0:
                    chapter_text = chapter_data['content']['textos'][0]['text']
                    
                    texts.append(chapter_text)
                    metadatas.append({
                        'chapter': chapter_title,
                        'chapter_number': chapter_num,
                    })
                    
            logger.info(f"Carregados {len(texts)} textos de {len(chapters_data)} capítulos")
            return texts, metadatas
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo JSON: {e}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa o texto para português
        """
        # Remove texto padrão do livro
        text = text.replace("Voltar ao indice", "")
        text = text.replace("Manual de cuidados paliativos", "")
        
        # Remove caracteres especiais mantendo acentos e letras portuguesas
        text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', ' ', text, flags=re.IGNORECASE)
        
        # Normaliza espaços
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs e emails
        text = re.sub(r'http\S+|www.\S+|\S+@\S+', '', text)
        
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Divide o texto em chunks menores
        """
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_size = len(sentence.split())
            
            if not sentence:
                continue
                
            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                # Adiciona chunk atual se tiver tamanho mínimo
                if current_size >= self.chunk_overlap:
                    chunks.append('. '.join(current_chunk) + '.')
                
                # Inicia novo chunk com overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
        
        # Adiciona último chunk se tiver tamanho mínimo
        if current_size >= self.chunk_overlap:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32
         ):
        """
        Adiciona textos ao índice com pré-processamento e monitoramento
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Pré-processamento
        processed_texts = []
        processed_metadatas = []
        
        for text, metadata in zip(texts, metadatas):
            # Divide em chunks se texto for muito grande
            if len(text.split()) > self.chunk_size:
                chunks = self.chunk_text(text)
                for i, chunk in enumerate(chunks):
                    # Copia e atualiza metadados para cada chunk
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'chunk_size': len(chunk.split()),
                        'start_pos': i * (self.chunk_size - self.chunk_overlap),
                        'total_chunks': len(chunks)
                    })
                    
                    processed_texts.append(self.preprocess_text(chunk))
                    processed_metadatas.append(chunk_metadata)
            else:
                processed_texts.append(self.preprocess_text(text))
                processed_metadatas.append(metadata)

        
        # Verificação de limite
        future_size = self.index.ntotal + len(processed_texts)
        if future_size > self.max_vectors_warning:
            logger.warning(
                f"Adição ultrapassará limite de {self.max_vectors_warning:,} vetores. "
                f"Total após adição: {future_size:,}"
            )

        # Geração de embeddings em batches
        logger.info("Gerando embeddings...")
        embeddings = []
        for i in tqdm(range(0, len(processed_texts), batch_size)):
            batch_texts = processed_texts[i:i + batch_size]
            batch_embeddings = self.embeddings.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        faiss.normalize_L2(embeddings)
        
        # Adiciona ao índice
        self.index.add(embeddings)
        
        # Atualiza documentos
        for text, metadata, embedding in zip(processed_texts, processed_metadatas, embeddings):
            doc = Document(
                page_content=text,
                metadata=metadata,
                embedding=embedding
            )
            self.documents.append(doc)
        
        self._log_index_stats()

    def similarity_search_mmr(
        self,
        query: str,
        k: int = None,
        lambda_param: float = None,
        filter_chapter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca com MMR e sumarização automática
        """
        # Inicializa o summarizer
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

        k = k or self.top_k
        lambda_param = lambda_param or self.lambda_param

        # Gera embedding da query
        query_embedding = self.embeddings.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Busca inicial
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k * 2)
        selected_indices = []
        remaining_indices = list(indices[0])

        while len(selected_indices) < k and remaining_indices:
       
            best_score = -np.inf
            best_idx = -1
            best_remaining_idx = -1
            
            for i, idx in enumerate(remaining_indices):
                # Relevância
                relevance = scores[0][indices[0] == idx][0]
                
                # Diversidade
                if selected_indices:
                    selected_embeddings = np.vstack([
                        self.documents[idx].embedding 
                        for idx in selected_indices
                    ])
                    similarity_to_selected = np.max(
                        selected_embeddings @ self.documents[idx].embedding
                    )
                else:
                    similarity_to_selected = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - \
                           (1 - lambda_param) * similarity_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
                    best_remaining_idx = i
            
            if best_idx != -1:
                if filter_chapter and self.documents[best_idx].metadata.get('chapter') != filter_chapter:
                    remaining_indices.pop(best_remaining_idx)
                    continue
                selected_indices.append(best_idx)
                remaining_indices.pop(best_remaining_idx)
            else:
                break
        
        return [self.documents[idx] for idx in selected_indices]

    def _log_index_stats(self):
        """Registra estatísticas do índice"""
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "total_documents": len(self.documents),
            "memory_usage_mb": (self.index.ntotal * self.dimension * 4) / (1024 * 1024)
        }
        
        logger.info(
            f"\nEstatísticas do Índice:\n"
            f"- Vetores: {stats['total_vectors']:,}\n"
            f"- Documentos: {stats['total_documents']:,}\n"
            f"- Memória: {stats['memory_usage_mb']:.2f} MB"
        )

    def save_vector_store(self, folder_path: str):
        """Salva a vector store com timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{folder_path}_{timestamp}"
        os.makedirs(save_path, exist_ok=True)
        
        # Salva índice
        faiss.write_index(self.index, f"{save_path}/index.faiss")
        
        # Salva configurações e documentos
        store_data = {
            'documents': [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'embedding': doc.embedding.tolist() if doc.embedding is not None else None  # Salva embedding
                }
                for doc in self.documents
            ],
            'config': {
                'embedding_model': "neuralmind/bert-base-portuguese-cased",
                'lambda_param': self.lambda_param,
                'top_k': self.top_k,
                'max_vectors_warning': self.max_vectors_warning,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'dimension': self.dimension
            }
        }
        
        with open(f"{save_path}/store_data.json", 'w', encoding='utf-8') as f:
            json.dump(store_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Vector store salva em: {save_path}")
    
    @classmethod
    def load_vector_store(cls, folder_path: str) -> 'VectorStoreFlatMMR':
        """Carrega vector store"""
        
        folder_path = Path(folder_path)  # Garante que o caminho seja um objeto Path
        
        # Carrega os dados de configuração e documentos do arquivo JSON
        with open(folder_path / "store_data.json", 'r', encoding='utf-8') as f:
            store_data = json.load(f)
        
        # Obtém a configuração armazenada
        config = store_data['config']
        
        # Cria a instância da vector store com as configurações carregadas
        store = cls(
            embedding_model=config.get('embedding_model', "neuralmind/bert-base-portuguese-cased"),
            lambda_param=config.get('lambda_param', 0.7),
            top_k=config.get('top_k', 5),
            max_vectors_warning=config.get('max_vectors_warning', 100000),
            chunk_size=config.get('chunk_size', 1000),
            chunk_overlap=config.get('chunk_overlap', 200)
        )
        
        # Carrega os documentos e seus embeddings
        for doc_data in store_data['documents']:
            embedding = np.array(doc_data['embedding']) if doc_data.get('embedding') is not None else None
            doc = Document(
                page_content=doc_data['text'],
                metadata=doc_data['metadata'],
                embedding=embedding  # Carrega o embedding, se houver
            )
            store.documents.append(doc)

        # Carrega o índice FAISS
        store.index = faiss.read_index(str(folder_path / "index.faiss"))
        
        logger.info(
            f"Vector store carregada de: {folder_path}\n"
            f"Total de documentos: {len(store.documents)}"
        )
        
        return store
    

def main():
    try:
        # Inicializa a vector store com configurações otimizadas
        store = VectorStoreFlatMMR(
            embedding_model="neuralmind/bert-base-portuguese-cased",
            lambda_param=0.7,
            chunk_size=500,  # Reduzido para maior granularidade
            chunk_overlap=100  # Ajustado proporcionalmente
            )

        logger.info("Iniciando processamento da vector store...")
        
        base_save_path = Path(__file__).parent / "app" / "faiss_index" / "index.faiss"
        
        texts, metadatas = load_texts_from_json(filepath)
        store.add_texts(texts, metadatas)
        
        # Salva e captura o caminho completo com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_save_path = f"{base_save_path}_{timestamp}"
        store.save_vector_store(base_save_path)  # isto criará o diretório com timestamp
        
        # Lista os diretórios e pega o mais recente
        save_dirs = [d for d in os.listdir(os.path.dirname(base_save_path)) 
                    if d.startswith(os.path.basename(base_save_path))]
        latest_dir = max(save_dirs)
        latest_path = os.path.join(os.path.dirname(base_save_path), latest_dir)
        
        # Carrega a versão mais recente
        logger.info(f"Carregando vector store mais recente de: {latest_path}")
        loaded_store = VectorStoreFlatMMR.load_vector_store(latest_path)
        
        print("\nInformações da Vector Store carregada:")
        print(loaded_store.get_index_info())
        
        test_query = "O que são cuidados paliativos?"
        logger.info("\nRealizando busca de teste...")
        results = store.similarity_search_mmr(test_query)

        # Inicializa o summarizer
        summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
        
        # Aplica sumarização aos resultados
        for i, doc in enumerate(results, 1):
            summary = summarizer(doc.page_content, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
            print(f"\nResultado {i}:")
            print(f"Capítulo: {doc.metadata.get('chapter', 'Capítulo não especificado')}")
            print(f"Resumo: {summary}")
            print(f"Texto Completo: {doc.page_content[:200]}...\n")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise

if __name__ == "__main__":
    main()      
