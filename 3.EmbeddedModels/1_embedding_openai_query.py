from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model_name = 'text-embedding-3-large', dimension = 32)

result = embedding.embed_query("What is the capital of india")

print(str(result))