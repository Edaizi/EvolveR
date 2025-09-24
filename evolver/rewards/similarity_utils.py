import numpy as np
from openai import OpenAI
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from tools.logger_factory import setup_logger
logger = setup_logger("similarity_utils")

class EmbeddingClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingClient, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1", api_key: str = "empty", model_name: str = "bge_m3"):
        if not hasattr(self, 'client'):  # Ensure __init__ runs only once for the singleton
            self.api_url = api_url
            self.api_key = api_key if api_key is not None else "empty"
            self.model_name = model_name
            logger.info(f"Initializing EmbeddingClient. Provider: {self.api_url}, Model: {self.model_name}")

            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_url,
                    timeout=20.0, # Add a reasonable timeout
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        self.max_retries = 3
        self.retry_delay_seconds=5

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fetches embeddings for a list of texts."""
        if not self.client:
            logger.error("EmbeddingClient is not initialized.")
            return np.array([])
        
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            # Return empty array for empty or invalid input to avoid API errors.
            return np.array([])

        try:
            embedding_obj = self.client.embeddings.create(
                input=texts,
                model=self.model_name,
            )
            embeddings = [d.embedding for d in embedding_obj.data]
            return np.array(embeddings)

        except Exception as e:
            logger.error(f"Error calling embedding API or parsing the response: {e}")
            return np.array([])
    

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two 1D vectors using scikit-learn
    for robustness and optimization.
    """
    if v1.ndim == 1:
        v1 = v1.reshape(1, -1)
    if v2.ndim == 1:
        v2 = v2.reshape(1, -1)
    
    if v1.shape[1] != v2.shape[1] or v1.size == 0:
        return 0.0
    
    # sklearn's cosine_similarity returns a 2D array (matrix), so we get the single value
    return sk_cosine_similarity(v1, v2)[0, 0]

def get_similarity_score(
    text1: Union[str, List[str]], 
    text2: Union[str, List[str]],
    embedding_client: EmbeddingClient
) -> float:
    """
    Calculates the maximum cosine similarity between two texts or lists of texts.
    - If both are strings, computes similarity between them.
    - If one is a string and the other is a list, computes the max similarity of the string against items in the list.
    - If both are lists, computes the max similarity between any pair of items from the two lists.
    """
    if not text1 or not text2 or not embedding_client:
        return 0.0

    list1 = [text1] if isinstance(text1, str) else text1
    list2 = [text2] if isinstance(text2, str) else text2

    # Flatten the list of texts to make a single API call for efficiency
    all_texts = list(set(list1 + list2))
    if not all_texts:
        return 0.0

    embeddings_array = embedding_client.get_embeddings(all_texts)
    
    if embeddings_array.size == 0:
        return 0.0
    
    # Create a mapping from text to its embedding vector
    embedding_map = {text: vec for text, vec in zip(all_texts, embeddings_array)}

    max_similarity = 0.0
    for t1 in list1:
        for t2 in list2:
            vec1 = embedding_map.get(t1)
            vec2 = embedding_map.get(t2)
            if vec1 is not None and vec2 is not None:
                similarity = cosine_similarity(vec1, vec2)
                if similarity > max_similarity:
                    max_similarity = similarity
                    
    return max_similarity 
