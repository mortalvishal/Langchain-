from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "delhi is capital of india",
    "Kolkata is capital of west bengal",
    "paris is capital of france",
]

result = embedding.embed_documents(documents)

print (str(result))