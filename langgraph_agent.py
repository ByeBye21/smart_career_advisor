from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from bs4 import BeautifulSoup
import os
import re

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation history"]
    next: str
    retrieved_answers: List[str]

# Load LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key= ""             # Write your API key here
)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load persisted Chroma vector store
vectorstore = Chroma(
    collection_name='career_advice',
    embedding_function=embedding_model,
    persist_directory='./chroma_db'
)

# Clean HTML and whitespace
def strip_html(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'&[^;]+;|[<>{}]|\s+', ' ', text).strip()
    return text

# Rephrase query for clarity and relevance
def rephrase_query(query: str, history: List[dict]) -> str:
    history_str = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history if msg['role'] in ['user', 'assistant']
    ])
    prompt = f"""
Previous conversation:
{history_str}

Rephrase the following question to make it clear, concise, and relevant to career advice. If it's unclear or not career-related, ask a clarifying question instead:

'{query}'
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return strip_html(response.content.strip())

# Synthesize final response
def synthesize_response(query: str, answers: List[str], history: List[dict]) -> str:
    cleaned = [strip_html(ans) for ans in answers if ans]
    context = ("\n".join([f"- {ans[:200]}" for ans in cleaned])
               if cleaned else "No relevant context found.")
    history_str = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history if msg['role'] in ['user', 'assistant']
    ])
    prompt = f"""
You are a career advisor AI. The user asked: '{query}'.

Previous conversation:
{history_str}

Retrieved context:
{context}

Respond with clear, relevant career advice. If it's off-topic or unclear, ask a clarifying question to steer it toward a career-related discussion. Offer actionable suggestions and resources.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return strip_html(response.content.strip())

# Nodes
def start_node(state: AgentState) -> AgentState:
    state['messages'] = state.get('messages', [])
    state['next'] = "input"
    return state

def input_node(state: AgentState) -> AgentState:
    if state['messages'] and isinstance(state['messages'][-1], HumanMessage):
        state['next'] = "decide"
    else:
        state['next'] = "error"
        state['messages'].append(AIMessage(content="Please enter a valid career-related question."))
    return state

def decide_node(state: AgentState) -> AgentState:
    query = state['messages'][-1].content
    prompt = f"""
You are helping users with career advice.

Given this user input: '{query}'

Determine the next step. Respond with:
- 'retrieve' if it is clearly a career-related question
- 'ask' if it is vague, unrelated, or needs clarification
"""
    decision = llm.invoke([HumanMessage(content=prompt)]).content.lower()
    state['next'] = "retrieve" if "retrieve" in decision else "clarify"
    return state

def retrieve_node(state: AgentState) -> AgentState:
    query = state['messages'][-1].content
    history = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in state['messages'][:-1]
    ]
    rephrased_query = rephrase_query(query, history)
    docs = vectorstore.similarity_search(rephrased_query, k=3)
    state['retrieved_answers'] = [doc.metadata.get('answers_body', '') for doc in docs]
    state['next'] = "respond"
    return state

def respond_node(state: AgentState) -> AgentState:
    query = state['messages'][-1].content
    answers = state.get('retrieved_answers', [])
    history = [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in state['messages'][:-1]
    ]
    response = synthesize_response(query, answers, history)
    state['messages'].append(AIMessage(content=response))
    state['next'] = "end"
    if 'retrieved_answers' in state:
        del state['retrieved_answers']
    return state

def clarify_node(state: AgentState) -> AgentState:
    query = state['messages'][-1].content
    response = f"I'm here to help with career questions. Could you clarify how this relates to your career or professional goals: '{query}'?"
    state['messages'].append(AIMessage(content=response))
    state['next'] = "end"
    return state

def error_node(state: AgentState) -> AgentState:
    state['messages'].append(AIMessage(content="I didn't understand that. Can you rephrase it with a focus on career or jobs?"))
    state['next'] = "input"
    return state

def end_node(state: AgentState) -> AgentState:
    return state

# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("start", start_node)
graph.add_node("input", input_node)
graph.add_node("decide", decide_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("respond", respond_node)
graph.add_node("clarify", clarify_node)
graph.add_node("error", error_node)
graph.add_node("end", end_node)

graph.set_entry_point("start")
graph.add_edge("start", "input")
graph.add_conditional_edges("input", lambda s: s['next'], {
    "decide": "decide",
    "error": "error"
})
graph.add_conditional_edges("decide", lambda s: s['next'], {
    "retrieve": "retrieve",
    "clarify": "clarify"
})
graph.add_edge("retrieve", "respond")
graph.add_edge("respond", "end")
graph.add_edge("clarify", "end")
graph.add_edge("error", "input")
graph.add_edge("end", END)

# Checkpoint and compile
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Main invocation method
def run_agent(query: str, thread_id: str, history: List[dict]) -> List[dict]:
    messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user"
        else AIMessage(content=msg["content"])
        for msg in history
    ]
    messages.append(HumanMessage(content=query))
    state = app.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    )
    return [
        {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
        for msg in state['messages']
    ]
