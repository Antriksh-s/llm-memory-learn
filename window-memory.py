import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import trim_messages, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# 1. Initialize the Model
llm = ChatOpenAI(model="gpt-4o-mini")

# 2. Define the Trimmer (The "Window" Logic)
# strategy="last" ensures we keep the most recent messages.
# token_counter=len with max_tokens=2 acts as a "Message Window" of 2.
trimmer = trim_messages(
    max_tokens=2,
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# 3. Define the Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use short answers."),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}"),
])

# 4. Build the Chain with a Pre-processing Step
# We use RunnablePassthrough.assign to trim the history BEFORE it hits the prompt
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: trimmer.invoke(x["chat_history"])
    )
    | prompt 
    | llm
)

# 5. Session Management
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 6. Wrap with History Logic
wrapped_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

'''
# --- EXECUTION ---
config = {"configurable": {"session_id": "window_demo"}}

print("--- Step 1: Telling the AI a fact ---")
resp1 = wrapped_chain.invoke({"input": "My favorite color is Aquamarine."}, config=config)
print(f"AI: {resp1.content}")

print("\n--- Step 2: Asking a follow-up (Still in window) ---")
resp2 = wrapped_chain.invoke({"input": "What is it?"}, config=config)
print(f"AI: {resp2.content}")

print("\n--- Step 3: Pushing the first fact out of the window ---")
# By now, the window of 2 messages is full. 
# The next message will push "My favorite color is Aquamarine" out.
wrapped_chain.invoke({"input": "I live in Tokyo."}, config=config)

print("\n--- Step 4: Testing the memory ---")
resp4 = wrapped_chain.invoke({"input": "Do you remember my favorite color?"}, config=config)
print(f"AI: {resp4.content}")
'''

# 5. Usage
config = {"configurable": {"session_id": "window_demo"}}

resp1 = wrapped_chain.invoke({"input": "What is the capital of my favorite country, France?"}, config=config)
print(resp1.content)

resp2 = wrapped_chain.invoke({"input": "What is the population of that city?"}, config=config)
print(resp2.content)

resp3 = wrapped_chain.invoke({"input": "What is the currency used there?"}, config=config)
print(resp3.content)

resp4 = wrapped_chain.invoke({"input": "Do you know which country is my favorite?"}, config=config)
print(resp4.content)