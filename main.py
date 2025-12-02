
import os
import uuid
import tempfile
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from subagents import search_query, weather_info
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from tools import rag_pdf, search_web, get_weather

st.set_page_config(page_title="Just an agent", layout='wide')
st.title('Simple Ollama Agent')

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'Howdy! What can I help you today?'}]

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'rag_tool' not in st.session_state:
    st.session_state.rag_tool = None

def current_agent():
    
    toolbox = [search_web, get_weather]

    if 'rag_tool' in st.session_state and st.session_state.rag_tool:
        toolbox.append(st.session_state.rag_tool)

    llm = ChatOllama(model='qwen2.5:7b-instruct', temperature=0.7)
    agent = create_agent(
        model=llm,
        tools=toolbox,
        middleware=[SummarizationMiddleware(
            model=llm, max_tokens_before_summary=2000, messages_to_keep=5
        )],
        checkpointer=InMemorySaver()
    )

    return agent

upload_file = st.file_uploader('Upload your pdf here', type='pdf', accept_multiple_files=False)

if upload_file and st.button('Process PDF'):
    with st.spinner('Processing...'):
        tool_desc = f'Use this tool to answer question about uploaded file: {upload_file.name}'
        new_tool = rag_pdf(upload_file, query=tool_desc)
        st.session_state.rag_tool = new_tool
        st.success(f'{upload_file.name} uploaded successfully!')

agent = current_agent()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if prompt := st.chat_input('Ask your research question...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)
    config = RunnableConfig(configurable = {"thread_id": st.session_state.thread_id})
    with st.chat_message('assistant'):
        with st.spinner('Thinking and running...'):
            try:
                response = agent.invoke({
                'messages':[{'role': 'user',
                'content': prompt}]}, config)
                agent_response = response['messages'][-1].content
                st.markdown(agent_response)

            except Exception as e:
                agent_response = f'An error occured: {e}'
                st.error(agent_response)

    st.session_state.messages.append({'role': 'assistant', 'content': agent_response})