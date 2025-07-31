import streamlit as st
import requests

st.title("🦜🔗 LAFISE On")

with st.form("my_form"):
    text = st.text_area("Preguntame! Soy LAFISE On:",)
    if submitted := st.form_submit_button("Submit"):
        with st.spinner("Waiting response..."):
            url = "http://192.168.2.2:5000/feedback"
            response = requests.post(url, json={"data": text})
            data = response.json()#.get("response", "No response")
            st.write(data)