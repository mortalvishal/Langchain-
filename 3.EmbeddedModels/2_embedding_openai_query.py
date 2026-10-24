from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model_name = 'text-embedding-3-large', dimension = 32)

documents = [
    "delhi is capital of india",
    "Kolkata is capital of west bengal",
    "paris is capital of france",
]

result = embedding.embed_documents(documents)

print(str(result))