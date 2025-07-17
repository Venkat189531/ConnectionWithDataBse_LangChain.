import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load env
load_dotenv()
groq_key = os.getenv("GROQ_KEY")

# LLM setup
llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192", streaming=True)

# Streamlit page config
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Sidebar inputs
mysql_host = st.sidebar.text_input("MySQL Host", value="localhost") 
mysql_user = st.sidebar.text_input("MySQL User")
mysql_password = st.sidebar.text_input("MySQL Password", type="password")
mysql_db = st.sidebar.text_input("MySQL Database")

# Connect DB
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}",
    sample_rows_in_table_info=3
)

# Toolkit and agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Initialize messages
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state["messages"]:
    st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

# Input box for user query
user_query = st.text_input("Your question:", key="user_input")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.markdown(f"**User:** {user_query}")

    with st.container():
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query)  # Optionally add callbacks=[streamlit_callback]
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.markdown(f"**Assistant:** {response}")
