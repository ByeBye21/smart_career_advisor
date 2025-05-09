import streamlit as st
from langgraph_agent import run_agent

# Streamlit app configuration
st.set_page_config(page_title="Smart Career Advisor", page_icon="💼", layout="wide")
st.title("Smart Career Advisor")
st.markdown("Ask career-related questions and get personalized advice powered by real data!")

# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_123"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input(placeholder="e.g., What careers are good for computer science majors?")

# Process query
if user_input:
    with st.spinner("Finding the best advice for you..."):
        try:
            # Append user message to history
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Run agent with full history
            messages = run_agent(user_input, st.session_state.thread_id, st.session_state.messages[:-1])
            # Update history with new messages, avoiding duplicates
            for msg in messages:
                if not any(m["role"] == msg["role"] and m["content"] == msg["content"] for m in st.session_state.messages):
                    st.session_state.messages.append(msg)
            # Rerun to display new messages
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Initial message
if not st.session_state.messages:
    st.info("Ask a career-related question to begin!")