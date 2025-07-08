from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from vector_store import VectorStore
from typing import Optional
import os
from datetime import datetime

class ChatManager:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # Initialize LangChain components
        # Note: You'll need to set OPENAI_API_KEY environment variable
        self.llm = OpenAI(
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        )
        
        # Create prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "history", "question"],
            template="""
            You are a helpful AI assistant with access to previous conversations.
            
            Context from similar past conversations:
            {context}
            
            Recent conversation history:
            {history}
            
            Human: {question}
            Assistant: """
        )
        
        # Initialize LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            verbose=True
        )
        
        # Session memories
        self.session_memories = {}
    
    def _get_session_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a session"""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ConversationBufferWindowMemory(
                k=5,  # Keep last 5 exchanges
                return_messages=True
            )
        return self.session_memories[session_id]
    
    def _build_context_from_similar_conversations(self, query: str, limit: int = 3) -> str:
        """Build context string from similar past conversations"""
        similar_conversations = self.vector_store.search_similar_conversations(query, limit)
        
        if not similar_conversations:
            return "No relevant past conversations found."
        
        context_parts = []
        for score, metadata in similar_conversations:
            if score > 0.5:  # Only include reasonably similar conversations
                context_parts.append(
                    f"Previous conversation (similarity: {score:.2f}):\n"
                    f"User: {metadata['user_message']}\n"
                    f"Bot: {metadata['bot_response']}\n"
                )
        
        return "\n".join(context_parts) if context_parts else "No highly relevant past conversations found."
    
    async def process_message(self, user_message: str, session_id: str) -> str:
        """Process user message and generate response"""
        try:
            # Get session memory
            memory = self._get_session_memory(session_id)
            
            # Build context from similar conversations
            context = self._build_context_from_similar_conversations(user_message)
            
            # Get recent conversation history
            history_items = self.vector_store.get_conversation_history(session_id, limit=3)
            history = "\n".join([
                f"User: {item['user_message']}\nBot: {item['bot_response']}"
                for item in history_items
            ]) if history_items else "No recent history."
            
            # Generate response using LangChain
            response = await self._generate_response(context, history, user_message)
            
            # Store conversation in vector store
            self.vector_store.add_conversation(user_message, response, session_id)
            
            # Update session memory
            memory.chat_memory.add_user_message(user_message)
            memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            # Still store the conversation even if there was an error
            self.vector_store.add_conversation(user_message, error_response, session_id)
            return error_response
    
    async def _generate_response(self, context: str, history: str, question: str) -> str:
        """Generate response using LangChain"""
        try:
            response = self.chain.run(
                context=context,
                history=history,
                question=question
            )
            return response.strip()
        except Exception as e:
            return f"I'm having trouble generating a response right now. Error: {str(e)}"
    
    def get_session_stats(self, session_id: str) -> dict:
        """Get statistics for a specific session"""
        history = self.vector_store.get_conversation_history(session_id, limit=1000)
        return {
            "session_id": session_id,
            "total_messages": len(history),
            "first_message": history[0]["timestamp"] if history else None,
            "last_message": history[-1]["timestamp"] if history else None
        }