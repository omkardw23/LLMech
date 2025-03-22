import torch
import ollama
from utils.logger import logger
import time
from typing import List
import asyncio
from functools import lru_cache

class EmbeddingService:
    def __init__(self):
        self.vault_content = []
        self.vault_embeddings_tensor = None
        self._embedding_cache = {}
        self.batch_size = 10  # Process 10 items at a time
        self.rate_limit_delay = 0.1  # 100ms delay between batches

    @lru_cache(maxsize=1000)
    def _get_embedding(self, text: str) -> List[float]:
        """Cache embeddings to avoid redundant API calls"""
        try:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None

    def initialize_embeddings(self):
        try:
            # Read vault content
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                self.vault_content = [line.strip() for line in vault_file.readlines() if line.strip()]

            if not self.vault_content:
                logger.warning("No content found in vault.txt")
                return False

            # Process embeddings in batches
            vault_embeddings = []
            total_items = len(self.vault_content)
            
            logger.info(f"Processing {total_items} items from vault...")
            
            for i in range(0, total_items, self.batch_size):
                batch = self.vault_content[i:i + self.batch_size]
                batch_embeddings = []
                
                for text in batch:
                    if text in self._embedding_cache:
                        embedding = self._embedding_cache[text]
                    else:
                        embedding = self._get_embedding(text)
                        self._embedding_cache[text] = embedding
                    
                    if embedding:
                        batch_embeddings.append(embedding)
                
                vault_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                if i + self.batch_size < total_items:
                    time.sleep(self.rate_limit_delay)

            if vault_embeddings:
                self.vault_embeddings_tensor = torch.tensor(vault_embeddings)
                logger.info(f"Successfully processed {len(vault_embeddings)} embeddings")
                return True
            else:
                logger.error("No embeddings were generated")
                return False

        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            return False

    def get_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Get relevant context from the vault based on similarity"""
        if not self.vault_embeddings_tensor or self.vault_embeddings_tensor.nelement() == 0:
            logger.warning("No embeddings available for context retrieval")
            return []

        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return []

            input_tensor = torch.tensor(query_embedding)

            # Ensure dimensions match
            if input_tensor.shape[0] != self.vault_embeddings_tensor.shape[1]:
                if input_tensor.shape[0] < self.vault_embeddings_tensor.shape[1]:
                    padding = torch.zeros(self.vault_embeddings_tensor.shape[1] - input_tensor.shape[0])
                    input_tensor = torch.cat([input_tensor, padding])
                else:
                    input_tensor = input_tensor[:self.vault_embeddings_tensor.shape[1]]

            # Calculate similarities
            cos_scores = torch.cosine_similarity(input_tensor.unsqueeze(0), self.vault_embeddings_tensor)
            top_k = min(top_k, len(cos_scores))
            top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
            
            return [self.vault_content[idx] for idx in top_indices]

        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return [] 