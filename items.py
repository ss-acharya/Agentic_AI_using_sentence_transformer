from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import re
import pickle
import os
import math
import json
import logging
import random
from tqdm import tqdm
import collections
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
from datasets import load_dataset
import chromadb
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import ollama
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7
DB = "products_vectorstore"
MAXIMUM_DATAPOINTS = 5_000
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['cyan', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'red']

class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """

    # This line is commented out as we don't directly use the tokenizer in this class
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    title: str
    price: float
    text: str
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        Clean up the details string by removing common text that doesn't add value
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also remove words that are 7+ chars and contain numbers, as these are likely irrelevant product numbers
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range,
        then set include to True
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price removed
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a String version of this Item
        """
        return f"<{self.title} = ${self.price}>"

        

    #with open('test.pkl', 'rb') as file:
    #    train = pickle.load(file)
        
   # print(f"There are {len(train):,} training items scraped from Amazon, and the first one is {train[0]}")

with open('train.pkl', 'rb') as file:
    train = pickle.load(file)
    
    
    
with open('test.pkl', 'rb') as file:
    test = pickle.load(file)

#print(train) 
#print(f"There are {len(train):,} training items scraped from Amazon, and the first one is {train[0]}")

client = chromadb.PersistentClient(path=DB)
  
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

vector = model.encode(["A room full of software engineers"])[0]
#vector = model.encode(["<Motorcraft YB3125 Fan Clutch = $225.11>"])[0]
print(vector.shape)
#vector


collection_name = "products_new"
existing_collection_names = [collection_.name for collection_ in client.list_collections()]
#collection = client.get_collection(name = "products")

#Full original code has been changed from (collection_name not in existing_collection_names)
if (collection_name not in existing_collection_names):
    collection = client.create_collection(collection_name)
    #for i in tqdm(range(0, len(train), 1000)):
    for i in tqdm(range(0, len(train), 1000)):
        documents = [item.text for item in train[i: i+1000]]
        vectors = model.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in train[i: i+1000]]
        ids = [f"doc_{j}" for j in range(i, i+1000)]
        collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
        )


else:
    collection = client.get_collection(name = "products")
#collection = client.get_or_create_collection(collection_name)
#print('The list of collection names:' + str(existing_collection_names))
#collection_new = collection

# Get the relevant collection
collection_new = ''

# Create the relevant collection (if needed)
if ("products_v3" not in existing_collection_names):
    collection_new = client.create_collection(name = "products_v3")
    
    for i in tqdm(range(len(train))):
        documents = [train[i].text]
        vectors = [model.encode(train[i].text).astype(float).tolist()]
        metadatas = [{'category': train[i].category, "price": train[i].price}]
        ids = [f"doc_{i}"]
        collection_new.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )
        
else:
    collection_new = client.get_collection(name = "products_v3")
    

result = collection_new.get(include=['embeddings', 'documents', 'metadatas'], limit=MAXIMUM_DATAPOINTS)
vectors = np.array(result['embeddings'])
documents = result['documents']
categories = [metadata['category'] for metadata in result['metadatas']]
colors = [COLORS[CATEGORIES.index(c)] for c in categories]
"""
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=4, color=colors, opacity=0.7),
    text=[f"Category: {c}<br>Text: {d[:50]}..." for c, d in zip(categories, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vectorstore Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y'),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
reduced_vectors = tsne.fit_transform(vectors)
# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=2, color=colors, opacity=0.7),
    text=[f"Category: {c}<br>Text: {d[:50]}..." for c, d in zip(categories, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=1200,
    height=800,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
"""

def make_context(similars, prices):
    message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
    for similar, price in zip(similars, prices):
        message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
    return message

def messages_for(item, similars, prices):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = make_context(similars, prices)
    user_prompt += "And now the question for you:\n\n"
    user_prompt += item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"}
    ]
    
def preprocess(item):
    system_message = "You rewrite product descriptions in a format most suitable for finding similar products in a Knowledge Base"
    user_message = "Please write a short 2-3 sentence description of the following product; your description will be used to find similar products so it should be comprehensive and only about the product. Details:\n"
    user_message += item
    user_message += "\n\nNow please reply only with the short description, with no introduction"
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    response = ollama.chat(
        model="llama2",
        messages=messages
    )
    #print("------*****2")
    #print(response)
    return response
    


"""
   
    response = ollama_via_openai.chat.completions.create(
        model="llama3.2",
        messages=messages,
        seed=42
    )
     return response.choices[0].message.content
"""
    

def vector(item):
    model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #vector = model1.encode(["<Motorcraft YB3125 Fan Clutch = $225.11>"])[0]
    #print(vector)
    
    text = preprocess(item.text)
    #print(item.text)
    #text = preprocess("Motorcraft YB3125 Fan Clutch")
    #print(text)
    #print(text['message']['content'])
    #print("------*****")
    #print(model1.encode((text['message']['content'])))
    #return model1.encode(text)[0]
    #return  model1.encode(item)
    return model1.encode((text['message']['content']))

def find_similars(item):
    #print("------*****3")
    #print(item)
    vec = vector(item)
    print(type(vec))
    #print("------*****2")
    #print(vec)
    results = collection_new.query(query_embeddings=vec.astype(float).tolist(), n_results=5)
    documents = results['documents'][0][:]
    prices = [m['price'] for m in results['metadatas'][0][:]]
    return documents, prices

print(test[10].text)

#print(preprocess(test[2].text))
documents, prices = find_similars(test[10])
#print("------")
print(documents, prices)
print(collection_new)
#print(train[0].price)






"""   
for i in range(len(test)):
    print('This is iteration ' + str(i + 1) + '.')
    documents, prices = find_similars(test[i])
    
    if (len(documents) > 0) and (len(prices)> 0):
        print('The index is ' + str(i) + '.')
        
        print(documents, prices)
        
        break

"""
#print(make_context(documents, prices))


