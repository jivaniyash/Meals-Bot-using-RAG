# streamlit run ./main.py
import streamlit as st

import chromadb
from chromadb.utils import embedding_functions 
import time
from PIL import Image

st.set_page_config(
    page_title="Chat with SLM", 
    page_icon="💬", 
    layout="centered", 
    initial_sidebar_state="auto", 
    menu_items=None
)

# Functions to cache model & db
# @st.cache_resource(show_spinner="Connecting to CSV doc...")
def load_csv():
    metadata = []
    docs = []
    ids = []

    with open("meals.csv", 'r') as file:
        import csv
        x = csv.DictReader(file)
        for i, row in enumerate(x):
            ids.append("id" + str(i+1))
            metadata.append({'category' : row['Category'],
                             'sub_category':row["Sub-Category"]})
            # docs.append(json.dumps({'Question' : row['Question'], 
            #              'Answer' : row['Answer']}))
            docs.append(row['Answer'])
    return ids, metadata, docs

@st.cache_resource(show_spinner="Loading Embeddings...")
def load_embedding_fn():
        # Initialize embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=False
        )
    
    return embedding_function

@st.cache_resource(show_spinner="Loading Gemma-2b-it model config...")
def load_model(model_path: str):
    # Load the pretrained model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

@st.cache_resource(show_spinner="Loading LLM pipeline...")
def load_pipeline(model_path:str):
    # Load the model pipeline
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_path)

    return pipe

# @st.cache_resource(show_spinner="Connecting to Database...")
def load_db(db_path: str):
    # Initialize chromadb client
    client = chromadb.PersistentClient(path=db_path)
    # client = chromadb.Client(database="Document_DB")
    
    return client

@st.cache_resource(show_spinner="Adding records to the collection...")
def load_collection():
    # Create or get the collection
    docs_collection = client.create_collection(
        name="docs_collection",
        metadata={"hnsw:space": "cosine"}, # l2 is the default
        get_or_create=True,
        embedding_function=embedding_function
        )
    docs_collection.add(
        documents=docs,
        metadatas=metadata,
        ids=ids
        )

    return docs_collection

# @st.cache_resource(show_spinner="Generating Answer...")
def retrieve_answer(prompt, use_pipeline=True):
    import time
    start_time = time.time()
    if use_pipeline:
        print(f"Generating output using pipeline...")
        llm_output = pipe(prompt_to_LLM, return_full_text=False, max_length=1024)
        generated_text = llm_output[0]["generated_text"]
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"Generating output using tokenizer & model object...")
        llm_output = model.generate(input_ids, 
                            max_length=512, 
                            num_return_sequences=1,  # always 1
                            # temperature=0.7,
                            # do_sample=True,
                            pad_token_id=model.config.eos_token_id,
                            )
        print(f"Tokenizing output...")
        generated_text = tokenizer.decode(token_ids=llm_output[0][input_ids.shape[1]+2:], # selecting only answer generated
                                        skip_special_tokens=True
                                        )
    end_time = time.time()
    print("Done... It took {} secs".format(round(end_time-start_time)))    
    return generated_text

# @st.cache_resource(show_spinner="Fetching Docs...")
def fetch_similar_docs(user_prompt:str, n_results=1):
    print(f"Fetching docs similar to - {user_prompt}")
    output = docs_collection.query(
        query_texts=[user_prompt],
        n_results=n_results
        )
    return output

# Load resources
ids, metadata, docs = load_csv()

model_path = "./gemma-2b-it"

use_pipeline = False

if use_pipeline:
    pipe = load_pipeline(model_path)
else:
    model, tokenizer = load_model(model_path)

embedding_function = load_embedding_fn()

db_path="./docs.db"
client = load_db(db_path=db_path)

docs_collection = load_collection()

def stream_data(text, sec=0.2):
    for word in text.split(" "):
        yield word + " "
        time.sleep(sec)

user_logo = "🧑‍💻"
assistant_logo = "🧑‍💻"

st.title("Meals Bot using RAG")
st.caption("🚀 Chatbot powered by Streamlit 🚀")

if "messages" not in st.session_state:
    # st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["messages"] = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if user_prompt := st.chat_input("Ask me something"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    st.chat_message("user", avatar=user_logo).write_stream(stream_data(user_prompt))
    # generated_answer = user_prompt + "////////////"
                  
    output = fetch_similar_docs(user_prompt, n_results=1)
    
    context = ''
    for doc in output['documents'][0]:
        context += doc 
    st.session_state.messages.append({"role": "assistant", "content": f"Here is the context found: \n{context}"})
    st.chat_message("assistant", avatar=assistant_logo).write_stream(stream_data(f"Here is the context found: \n{context}", sec=0.05))

    prompt_to_LLM = f'''Generate a response that answers the user's question using the information from the context and considering the inferred meal type. Ensure the answer adheres to the factual content within the context and avoids introducing irrelevant information. If you don't find any relevant information in the context, just say- "No. I cannot find the information you are looking for.".
Context:{context}
Question:{user_prompt}
'''
    with st.spinner("Generating Output-"):
        generated_answer = retrieve_answer(prompt=prompt_to_LLM, use_pipeline=use_pipeline)
    st.session_state.messages.append({"role": "assistant", "content": generated_answer})
    st.chat_message("assistant", avatar=assistant_logo).write_stream(stream_data(generated_answer))
else:
    st.warning("Please enter a question. I will try my best to retrieve answer.")