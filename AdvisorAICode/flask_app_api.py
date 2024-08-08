from openai import OpenAI
import os
import faiss
import pandas as pd
import numpy as np
import pickle
from bs4 import BeautifulSoup
from functools import cache
import time
import re
from flask import Flask, request, session, abort
#from flask_session import Session

# Set OpenAI API key
aiClient = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY'),  # this is also the default, it can be omitted
)

app = Flask(__name__)

app.config['SECRET_KEY'] = "zavidos_advisor"
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT']= False

#Session(app)

# Function to transform query to embedding
def query_to_embedding(query, embedding_model="text-embedding-3-large"):
    response = aiClient.embeddings.create(
        input=query,
        model=embedding_model
    )
    return np.array(response.data[0].embedding).astype('float32')

# Function to search for similar startups
def search_similar_startups(query_embedding, index, index_to_id_map, top_k=15):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    max_distance = np.max(distances)
    similarities = 1 - (distances / max_distance)

    return [(index_to_id_map[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

# Function to rewrite query for embedding using chat
def rewrite_query_for_embedding_chat(user_query, model="gpt-4-1106-preview"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to concisely rewrite queries for semantic search."},
        {"role": "user", "content": f"Please rewrite this query for a semantic search in a database of startup descriptions: '{user_query}'"}
    ]

    try:
        response = aiClient.chat.completions.create(
            model=model,
            messages=messages
        )
        #print(response)
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to generate response with GPT
def generate_response_with_gpt(startup_ids_with_scores, df, query, modello="gpt-3.5-turbo-1106"):
    startup_ids = [sid[0] for sid in startup_ids_with_scores]
    descriptions = (df[df['id'].isin(startup_ids)]['name'] + ": " + df[df['id'].isin(startup_ids)]['combined_description']).tolist()
    prompt = query + " Knowing the following startups descriptions: " + " ".join(descriptions)

    response = aiClient.chat.completions.create(
        model=modello,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You are helping the user find startups that can help them solve their problem using knowledge given by the user, reasoning on that and providing only relevant solutions and describe why the proposed startups solve the problem."},
            {"role": "system", "content": "When providing your answer always write startup names in bold using markdown syntax"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

@cache
def load_active_startups(active_startup_csv):
    return pd.read_csv(active_startup_csv)

@cache
def load_faiss_index(faiss_index_file):
    return faiss.read_index(faiss_index_file)

@cache
def load_embeddings():
    with open('embeddings_with_ids_large.pkl', 'rb') as f:
        return pickle.load(f)

def replace_bold_names_with_links(response, df):
    # Regular expression to find bold text in markdown (enclosed in double asterisks)
    pattern = re.compile(r"\*\*(.*?)\*\*")

    # Replace each found bold text with a markdown link if it's a startup name
    def replace_with_link(match):
        name = match.group(1)
        # Check if the name is in the DataFrame
        if name in df['name'].values:
            startup_id = df[df['name'] == name]['id'].iloc[0]
            url = f"https://app.retailhub.ai/profile/startup/{startup_id}"
            return f"<a href='{url}'>{name}</a>"
        else:
            return f"<b>{name}</b>"  # Keep the original text if not a startup name

    return pattern.sub(replace_with_link, response)

@app.route('/ask', methods=['POST'])
def main():
    
    # Initialize session state variables if they don't exist
    embeddings_with_ids_large = load_embeddings()
    index_to_id_map = {i: emb[0] for i, emb in enumerate(embeddings_with_ids_large)}
    index = load_faiss_index("faiss_index_large.index")
    active_startups = load_active_startups("active_startups.csv")

    input_json = request.get_json()
    #print(input_json)
    advisor_response = ""

    if input_json:  # Ensure there's a query to process
        advisor_response = process_query(input_json, embeddings_with_ids_large, index, index_to_id_map, active_startups)
        
    # Return the modified response
    return {
        "answer": advisor_response
    }


def process_query(input_json, embeddings_with_ids_large, index, index_to_id_map, active_startups):
    # Construct context for follow-up queries
    context = None
    if 'chat_history' in input_json:
        context = ""
        # For follow-up queries, build a context from the conversation history
        for item in input_json['chat_history']:
            context += f" {item['query']} {item['response']}"
    
    #print(context)

    # Combine query with context for rewriting, especially for follow-up queries
    if context is None:
        full_query = input_json['question']
    else:
        full_query = input_json['question'] + " " + context
    
    #print(full_query)
    
    rewritten_query = rewrite_query_for_embedding_chat(full_query)
    
    #print(rewritten_query)

    # Generate embedding and search for similar startups
    query_embedding = query_to_embedding(rewritten_query, embedding_model="text-embedding-3-large")
    similar_startups_ids_with_scores = search_similar_startups(query_embedding, index, index_to_id_map, top_k=15)

    # Generate a response from GPT based on the search results
    response = generate_response_with_gpt(similar_startups_ids_with_scores, active_startups, rewritten_query, modello="gpt-3.5-turbo-1106")

    # Replace startup names with clickable links in the response
    modified_response = replace_bold_names_with_links(response, active_startups)

    # Return the modified response
    return modified_response
