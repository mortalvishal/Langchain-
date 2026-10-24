from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np  

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

document = [
    "Artificial intelligence is transforming the way people work and communicate.",
    "Machine learning enables computers to learn patterns from data.",
    "Natural language processing helps computers understand human language.",
    "Deep learning uses neural networks to solve complex problems.",
    "Text embeddings represent sentences as numerical vectors that capture semantic meaning."
]

query = "What is the role of AI in modern technology?"

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)),key =lambda x:x[1])[-1]

print(query)
print(document[index])
print("similarity score is:",score)
