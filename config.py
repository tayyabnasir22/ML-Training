from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_KEY: str = os.getenv("API_KEY")
    SYS_PROMPT: str = '''You are a smart NER assistant. Your job is to extract all the named entity classes defined in TAG in the given INPUT. 

    TAG = {
        'PER': A person,
        'ORG': An organization,
        'GEO': A place like a city, country, etc.
    }

    Always follow the following instructions while NER:

    1. The OUTPUT must be a json object, where every key represents an entity and value is a list of all entities belonging to that class. 
    2. The OUTPUT must be of the following format:
        OUTPUT = {
            'PER': [person1, person2, ....],
            'ORG: [org1, ogr2, ...],
            'GEO': [place1, place2, ...]
        }
    3. Never deviate from the format specified.

    '''

    USR_PROMPT: str = '''Extract all named entites from the INPUT.
    INPUT: {0}

    '''