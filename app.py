from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain import  PromptTemplate
#from langchain.llms import CTransformers
#from langchain_community.llms import CTransformers
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers


from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.prompts import PromptTemplate


app = Flask(__name__)  #initializing flask application here in our code 

load_dotenv()

# Load API key and environment from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


#pc = Pinecone(api_key=PINECONE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


# Create the index instance
index_name = "medical-chatbot"  # Name of your Pinecone index
index = pc.Index(index_name)  # Create index instance using the Pinecone instance
embeddings = download_hugging_face_embeddings()


# Assuming embeddings is initialized correctly
docsearch = LangchainPinecone.from_existing_index(index_name=index_name,embedding=embeddings.embed_query)



PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

from ctransformers import AutoModelForCausalLM

#model = AutoModelForCausalLM.from_pretrained("models/llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama")


llm=CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)




@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)