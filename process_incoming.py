import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model":"bge-m3", 
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        #"model":"deepseek-r1", 
        "model":"llama3.2",
        "prompt": prompt,
        "stream": False
    })
        
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding'].shape))
similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
# print(similarities)
top_results=5
max_indx=similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df=df.loc[max_indx]
# print(new_df[['title','number','text']])

prompt = f''' Hey! I am teaching JavaScript in this course which based on challenge to take 90 seconds per video to teach. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[['title','number','start','end','text']].to_json(orient='records')}
-----------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, it's just for you) where and how much content is taught in which video (in which video and at what timestamp)and guide the user to go to that particular video. If user asks unrelated questions, tell him/her that you can only answer questions related to the course.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)['response']
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
 #   print(index, item["title"], item["number"], item["text"], item["start"], item["end"])




