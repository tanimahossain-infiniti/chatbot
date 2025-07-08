import streamlit as st
import websocket
import json
import threading
import time
from datetime import datetime
import uuid

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "ws" not in st.session_state:
    st.session_state.ws = None

class WebSocketClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.connected = False
        self.response_received = False
        self.last_response = ""
    
    def connect(self):
        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(self.url)
            self.connected = True
            return True
        except Exception as e:
            st.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    def send_message(self, message, session_id):
        if not self.connected:
            return None
        
        try:
            data = {
                "message": message,
                "session_id": session_id
            }
            self.ws.send(json.dumps(data))
            
            # Wait for response
            response = self.ws.recv()
            response_data = json.loads(response)
            return response_data.get("message", "No response")
            
        except Exception as e:
            st.error(f"Error sending message: {e}")
            return None
    
    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.connected = False

# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    
    # Connection status
    if st.session_state.ws_connected:
        st.success("✅ Connected to server")
    else:
        st.error("❌ Not connected")
    
    # WebSocket URL configuration
    ws_url = st.text_input(
        "WebSocket URL", 
        value="ws://localhost:8000/ws",
        help="Enter the WebSocket server URL"
    )
    
    # Connect/Disconnect button
    if st.button("Connect" if not st.session_state.ws_connected else "Disconnect"):
        if not st.session_state.ws_connected:
            # Connect
            client = WebSocketClient(ws_url)
            if client.connect():
                st.session_state.ws = client
                st.session_state.ws_connected = True
                st.rerun()
        else:
            # Disconnect
            if st.session_state.ws:
                st.session_state.ws.disconnect()
            st.session_state.ws_connected = False
            st.session_state.ws = None
            st.rerun()
    
    st.markdown("---")
    
    # Session info
    st.subheader("Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    # New session
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.title("AI Chatbot with Memory")
st.markdown("This chatbot remembers past conversations and can provide contextual responses based on previous interactions.")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input("What would you like to know?", disabled=not st.session_state.ws_connected):
    if not st.session_state.ws_connected:
        st.error("Please connect to the server first.")
    else:
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"*{timestamp}*")
        
        # Send message to server and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.ws.send_message(prompt, st.session_state.session_id)
            
            if response:
                st.markdown(response)
                response_timestamp = datetime.now().strftime("%H:%M:%S")
                st.caption(f"*{response_timestamp}*")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "timestamp": response_timestamp
                })
            else:
                st.error("Failed to get response from server.")

# Footer
st.markdown("---")
st.markdown(
    """
    **Features:**
    - 🧠 **Contextual Memory**: Remembers past conversations using vector embeddings
    - 🔍 **Similarity Search**: Finds relevant past interactions to inform responses
    - 💬 **Real-time Chat**: WebSocket-based communication for instant responses
    - 📊 **Session Management**: Separate conversation threads with unique session IDs
    """
)