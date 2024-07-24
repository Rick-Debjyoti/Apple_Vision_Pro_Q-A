import streamlit as st
from chatbot import get_response, get_sales_response

st.title("Apple Vision Pro Chatbot")

user_query = st.text_input("Ask me anything about Apple Vision Pro:")
user_persona = st.text_input("Describe your persona (optional):")

if user_query:
    if user_persona:
        response = get_sales_response(user_query, user_persona)
    else:
        response = get_response(user_query)
    st.write("Response:", response)