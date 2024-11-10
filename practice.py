import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config("Python Assistant", page_icon="🐍")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Python Programming Coding Assistant 🐍")
st.write("Hi, I am here to help you with your Python programming questions and tasks.")

# Sidebar setup
st.sidebar.title("GROQ LLM")
st.sidebar.write("Enter Your Groq API Key")

# API key input
key = st.sidebar.text_input("Enter Your API Key", type="password")

# Model selection
model_options = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]
selected_model = st.sidebar.selectbox("Select LLM Model", model_options, index=3)  # Default is llama-3.2-3b-preview

# Check if API key is provided
if not key:
    st.sidebar.warning("Please enter your API key to use the assistant.")
else:
    try:
        # Initialize the LLM with the selected model
        llm = ChatGroq(
            groq_api_key=key,
            model_name=selected_model
        )
    except Exception as e:
        st.error(f"Model '{selected_model}' does not exist. Please select a valid model.")
        st.stop()

# Function to get response from the selected model
def get_response(query, chat_history):
    template = """
You are a helpful Python coding assistant. Answer the following questions directly and accurately, focusing on solving the user's problem. Use the conversation history as context, but do not mention it explicitly in your response.

Chat history: {chat_history}

User question: {Input}
"""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "Input": query
    })

# Chat input and history display
user = st.chat_input("Enter Your Message")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

if user is not None and user != "":
    st.session_state.chat_history.append(HumanMessage(user))

    with st.chat_message("Human"):
        st.markdown(user)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user, st.session_state.chat_history))
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))
