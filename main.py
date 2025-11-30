import uuid
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from subagents import search_query, weather_info
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

llm = ChatOllama(model='qwen2.5:7b-instruct', temperature=0.7)


agent = create_agent(model=llm, 
                     tools=[search_query, weather_info],
                     middleware=[SummarizationMiddleware(model=llm, max_tokens_before_summary=2000, messages_to_keep=5)], 
                     checkpointer=InMemorySaver())

thread_id = str(uuid.uuid4())
config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

while True:
    query = input()
    if query.lower() in ['q', 'exit', 'quit']:
        break
    response = agent.invoke({
        'messages':[{'role': 'user',
                    'content': query}]
    }, config)

    print(response['messages'][-1].content) 