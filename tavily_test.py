import os
from tavily_test import TavilyClient
from dotenv import load_dotenv
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

client = TavilyClient(api_key=tavily_api_key)
response = client.search(
    query="what's the capital of Mongolia?",
    search_depth="advanced"
)
print(response)