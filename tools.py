import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch


load_dotenv()

def search_web(query: str) -> str:
    "Getting up-to-date information on the internet"

    print('running search web tool...')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

    engine = TavilySearch(max_results = 3, include_answer = True)
    try:
        answer = engine.invoke(query)
        return answer
    except Exception as e:
        return f'Error: {str(e)}'
    
@tool
def get_weather(city: str) -> str:
    "Getting current weather information"
    api_key = os.environ.get('OPENWEATHER_API_KEY')
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