from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def store_data_to_vectordb(text_content, index_name="faiss_index_chatbot"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, length_function=len)
    docs = text_splitter.create_documents([text_content])
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_name)
    
    print(f"Data stored successfully in {index_name}")

def retrieve_data_from_vectordb(query, index_name="faiss_index_chatbot"):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vectorstore.as_retriever(search_type="similarity")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    
    results = qa.invoke({"query": query})
    return results

def process_sample_file(query: str = "Marry had a little lamb"):
    loader = TextLoader("data/sample.txt", encoding="utf-8")
    documents = loader.load()
    
    text_content = documents[0].page_content
    
    store_data_to_vectordb(text_content)
    
    results = retrieve_data_from_vectordb(query, index_name="faiss_index_chatbot")
    print(f"Query: {results['query']}")
    print(f"Answer: {results['result']}")
    return results

# Example usage:
process_sample_file()