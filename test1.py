import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # what is meesage place holder ??
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory #in built memory in langchain --
from langchain_core.runnables.history import RunnableWithMessageHistory # wait for 15 mint
from langchain_core.messages import HumanMessage, AIMessage # by default your LLM try --> input --> human or AI meeage


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# 1. Define the Prompt with a Placeholder for history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{input}"),
])

# 2. Create the Chain
chain = prompt | llm

# 3. Manage History (dictionary to store sessions)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. Wrap the chain with history logic
wrapped_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 5. Usage
config = {"configurable": {"session_id": "user_1"}}

resp1 = wrapped_chain.invoke({"input": "What is the capital of France?"}, config=config)
print(resp1.content)

resp2 = wrapped_chain.invoke({"input": "What is the population of that city?"}, config=config)
print(resp2.content)