import os
import logging
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.schema import Document

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/chatbot.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

VECTOR_DB_TEXT_FILE = "data/vectordb.txt"

class ChatbotManager:
    def __init__(self, index_name="vector_index_chatbot"):
        self.index_name = index_name
        self.llm = OllamaLLM(model="llama3.2")
        self.embeddings = OllamaEmbeddings(model="llama3.2")

        self.conversations = {}
        self.vectorstore = None
        if os.path.exists(self.index_name):
            self.vectorstore = Chroma(
                persist_directory=self.index_name, 
                embedding_function=self.embeddings
            )

    def create_index(self, file_path=VECTOR_DB_TEXT_FILE):
        try:
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("Conversation History:\n")
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                docs, 
                self.embeddings, 
                persist_directory=self.index_name
            )
            self.vectorstore.persist()
            logging.info(f"Vector store created and saved as {self.index_name}")
        except Exception as e:
            logging.error(f"Error creating vector index: {str(e)}")

    def save_answer_to_vector_db(self, question: str, answer: str, session_id: str):
        """Save the AI's answer to the main vector database"""
        try:
            if self.vectorstore is None:
                return
            
            # Create a document with the question-answer pair
            doc_content = f"Question: {question}\nAnswer: {answer}"
            doc = Document(
                page_content=doc_content,
                metadata={"session_id": session_id, "type": "qa_pair", "source": "chatbot_response"}
            )
            with open(VECTOR_DB_TEXT_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n{doc_content}\n")
            
            # Add to existing vector store
            self.vectorstore.add_documents([doc])
            
            # Persist the updated vector store
            self.vectorstore.persist()
            logging.info(f"Answer saved to vector DB for session {session_id}")
        except Exception as e:
            logging.error(f"Error saving answer to vector DB: {str(e)}")

    def get_conversation_chain(self, session_id):
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please create the index first.")

        if session_id not in self.conversations:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retriever = self.vectorstore.as_retriever()
            
            qa_prompt = PromptTemplate(
                template="""You are a helpful and friendly AI assistant. You can answer questions, hold normal conversations, and remember what the user has told you in this session.

You also have access to external documents and chat history that you may use to enhance your answer **if they are clearly relevant**. If not, answer using your own general knowledge or conversation skills.

Always try to:
- Understand the intent behind short or vague inputs.
- Ask clarifying questions if needed.
- Keep the conversation engaging and natural.
- Use the chat history for personalization, like remembering names or interests.
- Never force connections with document context if they don't make sense.

---
Document Context (use only if clearly relevant): 
{context}

Human: {question}
Assistant:""",
                input_variables=["context", "question"]
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
            answer = result.get("answer", "Sorry, I could not find an answer.")
            
            self.save_answer_to_vector_db(query, answer, session_id)
            return answer
        except Exception as e:
            logging.error(f"Error during chat: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
