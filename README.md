# ML-Training-COMSATS

#### Link to Kaggle Notebooks:
[Kaggle Code](https://www.kaggle.com/tayyabnasir22/code)


### Setting things up on Mac/Linux:
- Install [Python 3.11](https://www.python.org/downloads/release/python-3110/).
- Navigate to the Repo folder:
    - cd Documents/Github/ML-Training-COMSATS (change the path according to your system)
- Create a new virtual environment:
    - python -m venv env
- Activate the environment:
    - source env/bin/activate
- Install all the requirements:
    - pip install -r requirements.txt
- Create an .env file in your folder and add to it the following:
    - API_KEY = "Insert Value Here" 
    - Update the value with your Together access key, or leave it as it is if you do not need to work with the LLM part of the code
- Download the model-vectorizer file from [Kaggle](https://www.kaggle.com/code/tayyabnasir22/ner-with-logisticregression/output), and place it in your repository.
- Run the Fast API server:
    - uvicorn main:app --reload
- In your browser open http://127.0.0.1:8000/docs#/ to get access to the API documentation.