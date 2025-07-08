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
    return {"response": f"{req.role} said: {req.message} -backend response"}