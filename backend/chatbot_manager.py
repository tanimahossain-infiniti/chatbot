import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv()

class ChatbotManager:
    def __init__(self, index_name="faiss_index_chatbot"):
        self.index_name = index_name
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="llama3.2")

        self.conversations = {}
        self.vectorstore = None
        if os.path.exists(self.index_name):
            self.vectorstore = FAISS.load_local(
                self.index_name, self.embeddings, allow_dangerous_deserialization=True
            )

    def create_index(self, file_path="data/sample.txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(self.index_name)
        print(f"Vector store created and saved as {self.index_name}")

    def get_conversation_chain(self, session_id):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please create the index first.")

        if session_id not in self.conversations:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever = self.vectorstore.as_retriever()
            
            qa_prompt = PromptTemplate(
                template="""You are a helpful AI assistant with broad knowledge. You can have normal conversations, answer general questions, and remember what users tell you. When relevant document context is provided, you can use it to enhance your answers, but you're not limited to only that context.

Document Context (use if relevant): {context}

Chat History: {chat_history}

Human: {question}
Assistant:""",
                input_variables=["context", "chat_history", "question"]
            )
            
            self.conversations[session_id] = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
        return self.conversations[session_id]

    def chat(self, session_id: str, query: str) -> str:
        try:
            chain = self.get_conversation_chain(session_id)
            result = chain.invoke({"question": query})
            return result.get("answer", "Sorry, I could not find an answer.")
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"