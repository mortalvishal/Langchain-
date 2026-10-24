from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model_name = 'gpt-4', temperature = 0.9, max_completion_tokens = 10, max_retries = 3, request_timeout = 30)

result = model.invoke("what is the capital of india")

print (result.content)