import os
import uuid
import yaml
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from tools import  search_web, get_weather, rag_pdf

os.environ["GROQ_API_KEY"] = st.secrets['GROQ_API_KEY']

def load_config():
    with open('prompt.yaml', 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()
sys_prompt = config['agent_system']

st.set_page_config(page_title="Just an agent", layout='wide')
st.title('Simple AI Agent')

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
    llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.5)
    # llm = ChatOllama(model='qwen2.5:7b-instruct', temperature=0.7)
    agent = create_agent(
        model=llm,
        tools=toolbox,
        middleware=[SummarizationMiddleware(
            model=llm, trigger=('tokens', 2000), keep=('messages', 5),
        )],
        checkpointer=InMemorySaver(),
        system_prompt=sys_prompt
    )

    return agent

upload_file = st.file_uploader('Upload your pdf here', type='pdf', accept_multiple_files=False)

if upload_file and st.button('Process PDF'):
    with st.spinner('Processing...'):
        try:
            new_tool = rag_pdf(upload_file)
            st.session_state.rag_tool = new_tool
            st.success(f'{upload_file.name} uploaded successfully!')
        except Exception as e:
            if not st.session_state.rag_tool:
                st.session_state.rag_tool = None
            st.error(f'Failed to process {upload_file.name}: {e}')

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
                response = response['messages'][-1].content
                if isinstance(response, list):
                    agent_response = "".join([item['text'] for item in response if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item])
                else:
                    agent_response = response
                st.markdown(agent_response)

            except Exception as e:
                agent_response = f'An error occured: {e}'
                st.error(agent_response)

    st.session_state.messages.append({'role': 'assistant', 'content': agent_response})
