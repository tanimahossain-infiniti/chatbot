from chatbot_manager import retrieve_data_from_vectordb, store_data_to_vectordb
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application!"}

class ChatRequest(BaseModel):
    role: str
    message: str

@app.post("/chat/")
def chat(req: ChatRequest):
    store_data_to_vectordb(req.message)
    response = retrieve_data_from_vectordb(req.message, index_name="faiss_index_chatbot")
    return {"response": f"{response['result']}"}