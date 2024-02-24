import joblib
import json


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
    words = sentence.split(' ')
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

# Functions that must be implemented

# This function must contain the logic for loading up the model that will be used for prediction
def model_fn(model_dir):
    local_file_path = f'{model_dir}/model_vectorizer.pkl'  # Specify the local file name

    with open(local_file_path, 'rb') as f:
        vectorizer, model = joblib.load(f)
    
    return vectorizer, model

# This function will recieve the prediction request
def predict_fn(input_data, model):
    # Preprocess the input data (if necessary)
    # Make predictions using the loaded model
    vectorizer, clf = model
    transformed = vectorizer.transform(tokenizer(input_data['input'][0]))
    predictions = clf.predict(transformed)
    return {'output': list(predictions)}

# The following functions are used for serializing/deserializing the input and output, for examples 

def input_fn(input_data, content_type):
    # Implement deserialization logic based on the content type
    if content_type == 'application/json':
        return json.loads(input_data)
    else:
        # Handle other content types if needed
        raise ValueError('Unsupported content type: {}'.format(content_type))

def output_fn(prediction_output, accept):
    # Implement serialization logic based on the accept header
    if accept == 'application/json':
        return json.dumps(prediction_output), 'application/json'
    else:
        # Handle other accept headers if needed
        raise ValueError('Unsupported accept header: {}'.format(accept))