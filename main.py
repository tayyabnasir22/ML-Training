from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import Settings
from TogetherCommunicator import TogetherCommunicator
import pickle
import spacy

settings = Settings()
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


with open('./model_vectorizer.pkl', 'rb') as f:
    vectorizer, clf = pickle.load(f)

NER = spacy.load("en_core_web_sm")

@app.get("/", tags=["HeartBeat"])
def HeartBeat():
    """
    Return Hello World object if API is up and running.
    """
    return {"Hello": "World"}


def GenerateFeaturesForSentence(word, prevWord, nextWord, pos):
    # A single feature per word
    return { 
        "lowercase": word.lower(),
        "prevword": prevWord,
        "nextword": nextWord,
        "iscaps": str(word.isupper()),
        "istitlecase": str(word.istitle()),
        "isdigit": str(word.isdigit()),
        "pos": pos,
       }

def tokenizer(sentence):
    examples = []
    print(sentence)
    words = sentence.split(' ')
    print(words)
    pos = [r.split('/')[1] for r in words]
    words = [r.split('/')[0] for r in words]
    for index, word in enumerate(words):
        if index == 0:
            prevWord = '<start>'
        else:
            prevWord = words[index - 1]
        if index + 1 < len(words):
            nextWord = words[index + 1]
        else:
            nextWord = '<end>'
        examples.append(GenerateFeaturesForSentence(word, prevWord, nextWord, pos[index]))
    return examples
# Barack/NNP Obama/NNP will/MD be/VB visiting/VBG Lahore/NNP Pakistan/NNP in/IN 2024/CD for/IN Chess/NNP competition/NN
@app.get("/ml-ner")
def GetMLNER(text: str):
    transformed = vectorizer.transform(tokenizer(text))
    predictions = clf.predict(transformed)
    return ' '.join(predictions)

# Barack Obama will be visiting Lahore Pakistan in 2024 for Chess competition
@app.get("/spacy")
def SpacyNER(text: str):
    tagged = NER(text)
    tags = []
    for j, word in enumerate(tagged):
        tags.append(str(word) + ': ' + str(word.ent_type_))

    return ' '.join(tags)

@app.get("/ner", )
def GetNER(text: str, model: str):
    togetherAPI = TogetherCommunicator(settings.API_KEY)

    messages = togetherAPI.get_messages(settings.SYS_PROMPT, settings.USR_PROMPT.format(text))

    if model == 'llama':
        return togetherAPI.together_chat_api(togetherAPI.generate_llama_v2_chat_prompt(messages), 'togethercomputer/llama-2-7b-chat')
    else:
        return togetherAPI.together_chat_api(togetherAPI.generate_mistral_chat_prompt(messages), 'mistralai/Mistral-7B-Instruct-v0.1')

