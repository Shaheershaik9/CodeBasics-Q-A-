import streamlit as st
from langchain_helper import get_QA_chain,create_vector_db

st.title("CodeBasics QA 🙋")

button = st.button("Create a Knowledgebase")

if button:
    pass

question = st.text_input("Question:  ")

if question:
    chain = get_QA_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result:"])