import openai
import streamlit as st
import pandas as pd
import pickle
import json
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

# Constants
EMBEDDING_MODEL = "EMBED"
COMPLETION_MODEL = "GPT_35"
openai.api_type = "azure"
openai.api_base = "https://azure-openai-mfec-test.openai.azure.com/"
openai.api_version = "2022-12-01"

# Read data to df
df = pd.read_csv("data_source/policy_dataset.csv")

# Establish a cache of embeddings to avoid recomputing
# Cache is a dict of tuples (text, model) -> embedding, saved as a pickle file

# Set path to embedding cache
embedding_cache_path = "recommendations_embeddings_cache.pkl"

# Load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# Define a function to retrieve embeddings from the cache if present, and otherwise request via the API
def embedding_from_string(string, model=EMBEDDING_MODEL, embedding_cache=embedding_cache):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

# Generate prompt function
def gen_prompt_recommendations_from_strings(source_dataframe, user_input, k_nearest_neighbors, model=EMBEDDING_MODEL):
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all item
    embeddings = df.loc[:,"embedding"].tolist()
    # get the embedding of the user_input
    query_embedding = embedding_from_string(user_input)
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    prompt = f"""#### Instruction ####
You are a telesales representative at an insurance company. Your role is to find suitable products for recommendation based on the customer profile with reference website . 
If there is no product that matches the customer's interests, do not make any recommendations. Do not mention any product, you wouldn't suggest.

#### Example ####

Customer_Profile: Customer is 20 year-old student who just buy his first car and don't have any car insurance.

Product_List:
    1. Young Driver's Guardian
    Category: Insurance
    Descriptions: Young Driver's Guardian is specifically designed for new and inexperienced drivers. This policy offers specialized coverage that helps young drivers build a positive driving history while providing protection against accidents and theft. Enjoy competitive rates and guidance tailored to the needs of young drivers.
    Reference: https://www.a.com
    
    2. Family Protection Bundle
    Category: Insurance
    Descriptions: The Family Protection Bundle is designed to provide comprehensive coverage for your entire family. This policy extends coverage to multiple vehicles and drivers, offering peace of mind for every member of your household. Simplify your insurance needs and protect your loved ones with this all-in-one package.
    Reference: https://www.b.com
    
    3. Comprehensive Shield
    Category: Insurance
    Descriptions: Comprehensive Shield provides extensive coverage for your vehicle, protecting you against theft, accidents, natural disasters, and more. With this policy, you can drive with peace of mind, knowing that you're covered in various scenarios.
    Reference: https://www.c.com

Recommendation: Young Driver's Guardian is the best product for you. It is specifically designed for new and inexperienced drivers. This policy offers specialized coverage that helps young drivers build a positive driving history while providing protection against accidents and theft. Enjoy competitive rates and guidance tailored to the needs of young drivers.
Reference: https://www.a.com
DONE!

#### Completion ####

Customer_Profile: {user_input}

Product_List:
    
"""
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        prompt += f"""{k_counter}. {df.loc[i,"policy_name_en"]}
        Category: {df.loc[i,"category_name_en"]}
        Descriptions: {df.loc[i,"policy_description_benefit"]}
        Reference: https://www.policy.com
        """
        ###Reference: {df.loc[i,"reference"]}\n
    #  ({k_counter} of {k_nearest_neighbors}) 
    # Distance: {distances[i]:0.3f}
    prompt += """Recommendation: """
    return prompt

# Response function
def response_user(user_input, k, temp, model=COMPLETION_MODEL):
    prompt = gen_prompt_recommendations_from_strings(
        source_dataframe=df,
        user_input= user_input,
        k_nearest_neighbors=k
        )
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        temperature=temp,
        max_tokens=500,
        top_p=0.5,
        frequency_penalty=0.2,
        presence_penalty=0.5,
        stop=["DONE!"]
        )
    return response.choices[0].text

### Streamlit
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # Handle k value
    k = st.text_input("Top-K", key="top_k_number", value=3)
    try:
        k = int(k)
        if (k == 0) or k > len(df):
            raise Exception('Invalid Value')
    except:
        st.info(f"Top-K should be a number between 1 and {len(df)}")
        st.stop()
    # Handle temp value 
    temp = st.text_input("Temperature", key="temperature", value=0.1)
    try:
        temp = float(temp)
        if not(0 <= temp <= 1):
            raise Exception('Invalid Value')
    except:
        st.info(f"Temperature should be a number between 0.0 and 1.0")
        st.stop()
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Recomendation Chatbot")
st.chat_message("assistant").write("Please insert user profile for recommendation.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    # Add embedding col to df
    df["embedding"] = df["policy_description_benefit"].apply(lambda x: embedding_from_string(x))
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = response_user(user_input, k, temp)
    msg = response.replace('Reference:','\nReference:')
    # st.session_state.messages.append(msg)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
