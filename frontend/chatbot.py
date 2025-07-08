import streamlit as st
import requests
st.title("GraphBit Chatbot")

role_avatar = {
    "assistant": "🦖",
    "user": "🧑‍💻",
}
with st.chat_message("assistant", avatar=role_avatar["assistant"]):
    st.write("Hello, I am your AI assistant. How can I help you today?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=role_avatar.get(message["role"], "🧑")):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    with st.chat_message("user", avatar=role_avatar["user"]):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        api_response = requests.post(
            "http://localhost:8000/chat/",
            json={"role": "user", "message": prompt},
            timeout=10
        )
        if api_response.ok:
            response = api_response.json().get("response", f"Echo: {prompt}")
        else:
            response = f"API Error: {api_response.status_code}"
    except Exception as e:
        response = f"Request failed: {e}"

    with st.chat_message("assistant", avatar=role_avatar["assistant"]):
        st.markdown(response)

    
    st.session_state.messages.append({"role": "assistant", "content": response})