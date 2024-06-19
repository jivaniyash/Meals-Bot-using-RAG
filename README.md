# Meals Bot using RAG

## Demo
![demo-gif](https://github.com/jivaniyash/Meals-Bot-using-RAG/blob/master/demo/demo_video.gif)

## Usage
This repository provides a chat bot assistant using the gemma-2b model powered by Streamlit. Follow the instructions below to set up and run the application. 

## Overview 
1. [meals.csv](https://github.com/jivaniyash/Meals-Bot-using-RAG/blob/master/meals.csv) file contains 11 sample records for different types of meals offered by one of the Airline - Air Astana. Records are retrieved from [special-meals](https://www.airastana.com/global-en/booking-manage/special-meals) and structured into csv format.
2. These csv documents are parsed and convert to text-embeddings using `Sentence Transformer` library.
3. These embeddings are stored in the chromadb in the form of documents.
4. User asks a question about the meal.
5. Sentence Transformers convert the Natural Language to embeddings and retreives the most relevant document using vector similarity search from the Chromadb. 
6. This retrieved document is passed as a `context` to the LLM (SLM-`gemma-2b-it`) model along with the user question to generate answer.
7. Generated answer is displayed in the front end. 

## Requirements
- Python 3.7 or higher
- virtualenv
- git
- git-lfs

## Steps to Start the App

### 1. Clone the Repository
```sh
git clone https://github.com/jivaniyash/meals-bot-using-RAG
```

### 2. Set up a Virtual Environment
```sh
virtualenv venv
source venv/bin/activate
```

### 3. Install Git LFS & Download Model
```sh
sudo apt-get install git-lfs
git clone https://huggingface.co/google/gemma-2b-it
```
It will ask for authorization if you haven't provided your consent in the Huggingface. Also, It will ask for user_name & password as `huggingface token`. Please ensure credentials are correct to download the model files locally. 

### 4. Install Python Requirements
```sh
pip install -r requirements.txt
```

### 5. Run the App
At first run, it will take time to load the files properly.
```sh
streamlit run ./main.py
```

Some of the sample questions to explore - 
1. What items are availble for diabetic patients?
2. Is vegan meal available?
3. Hello. I need veg meals. Can you please provide what food items are available?
4. What ages are limited for child meals?



