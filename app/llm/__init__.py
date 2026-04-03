from app.llm.embedding_model import FastEmbedEmbeddings
from app.llm.google_model import GoogleLLMModel

llm_chat = GoogleLLMModel()
embeddings = FastEmbedEmbeddings()

__all__ = ["llm_chat", "embeddings"]