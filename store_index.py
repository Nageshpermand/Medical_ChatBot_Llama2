#This code is used to store data vectors into pincone Database
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv
import os
#from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone

#from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone


# Create an instance of the Pinecone class

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()



pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index instance
index_name = "medical-chatbot"  # Name of your Pinecone index
index = pc.Index(index_name)  # Create index instance using the Pinecone instance
# Create the Pinecone vector store from texts
docsearch = LangchainPinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)