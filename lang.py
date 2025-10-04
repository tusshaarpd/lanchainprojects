import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

# 🔑 Put your API key here
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(page_title="LangChain Chat", page_icon="🤖")
st.title("🤖 LangChain Chatbot with Memory")

# ✅ Persist memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "conversation" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True
    )

# Chat input
user_input = st.text_input("You:", key="user_input")
if st.button("Send"):
    if user_input:
        response = st.session_state.conversation.predict(input=user_input)
        st.session_state["chat_history"] = st.session_state.get("chat_history", [])
        st.session_state["chat_history"].append(("You", user_input))
        st.session_state["chat_history"].append(("Bot", response))

# Display chat history
if "chat_history" in st.session_state:
    st.subheader("Chat History")
    for speaker, msg in st.session_state["chat_history"]:
        st.markdown(f"**{speaker}:** {msg}")
