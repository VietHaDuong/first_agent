import os
import requests
import tempfile
import streamlit as st
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

@tool
def search_web(query: str) -> str:
    """Use this tool to search the internet for current events, real-time data, 
    and up-to-date news. Useful for any query involving dates after 2024 
    or verified facts. Input should be a targeted search query.
    
    Arg:
        query: input from user to search for information (e.g. How old is Taylor Swift?)"""

    print('running search web tool...')
    if 'TAVILY_API_KEY' in st.secrets:
        os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
    else:
        st.error('Tavily api is not in st secret')
        st.stop()

    engine = TavilySearch(max_results = 3, include_answer = True)
    try:
        answer = engine.invoke(query)
        return answer
    except Exception as e:
        return f'Error: {str(e)}'
    
@tool
def get_weather(city: str) -> str:
    """    
    Use this when user wants to get the weather information of a city

    Arg:
        query: The full natural language request regarding weather (e.g., "What is the weather in Tokyo?")."""
    
    api_key = st.secrets['OPENWEATHER_API_KEY']
    base_curr = 'https://api.openweathermap.org/data/2.5/weather'
    base_geo = 'http://api.openweathermap.org/geo/1.0/direct'

    print('running weather tool...')
    try:
        geo_p = {'q': city, 'appid': api_key, 'limit': 1}
        geo = requests.get(base_geo, geo_p).json()

        lat = geo[0]['lat']
        lon = geo[0]['lon']

        curr_p = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}

        curr = requests.get(base_curr, curr_p).json()

        return curr
    
    except Exception as e:
        return f'Error: {str(e)}'
    
def rag_pdf(upload_file, query: str):
    """
    ONLY use this tool when the user asks about the 'uploaded file', 'the PDF', 
    'the context', or 'this document'. 
    Do NOT use this for general world knowledge or news.
    """
    os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(upload_file.getvalue())
            tmp_path = tmp_file.name
        loader =PyPDFLoader(tmp_path)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        text = text_splitter.split_documents(doc)
        embed = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
        vector = FAISS.from_documents(documents=text, embedding=embed)
        retriever = vector.as_retriever(search_kwargs={'k': 3})
        rag_tool = create_retriever_tool(retriever, description=query, name='pdf_rag')
        return rag_tool
    except Exception as e:
        return f'Error: {str(e)}'
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)