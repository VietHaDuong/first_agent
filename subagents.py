from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from tools import search_web,get_weather


llm = ChatOllama(model='qwen2.5:7b-instruct', temperature=0.7)

search_agent = create_agent(model=llm, tools=[search_web], system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information.")
weather_agent = create_agent(model=llm, tools=[get_weather], system_prompt="You are a helpful weather assistant. Use web search to get weather information.")

@tool
def search_query(query:str) -> str:
    """
    Use this tool for up-to-date information

    Arg:
        query: input from user to search for information (e.g. How old is Taylor Swift?)
    """
    response = search_agent.invoke({'messages':[{'role': 'user', 'content': query}]})
    return response['messages'][-1].content

def weather_info(query:str) -> str:
    """    
    Use this when user wants to get the weather information of a city

    Arg:
        query: The full natural language request regarding weather (e.g., "What is the weather in Tokyo?")."""
    response = weather_agent.invoke({'messages':[{'role': 'user', 'content': query}]})
    return response['messages'][-1].content