from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv  

load_dotenv()

model = ChatAnthropic(model_name = 'practice ', temperature = 0.9, max_completion_tokens = 10, max_retries = 3, request_timeout = 30)

result = model.invoke("what is the capital of india")

print (result.content)