import streamlit as st
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from PIL import Image


# import os
# from fastapi import FastAPI, Form
# from dotenv import load_dotenv
# from ibm_watsonx_ai import Credentials
# from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.foundation_models import ModelInference
# from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
# from model import Message, PromptMessage


st.set_page_config(
    page_title= "TALA Service"
)


html_css = """
    <style>
        @charset 'utf-8';

        html {
            font-family: 'Noto Sans KR', sans-serif;
            font-size: 14px;
        }

        p {
            line-height: 1.6;
        }

        .local-nav {
            position: fixed;
            top: 45px;
            left: 0;
            z-index: 11;
            width: 100%;
            height: 52px;
            padding: 0 1rem;
            border-bottom: 1px solid #ddd;
        }

        .local-nav-links {
            display: flex;
            align-items: center;
            max-width: 1000px;
            height: 100%;
            margin: 0 auto;
        }

        .global-nav-links {
            justify-content: space-between;
        }

        .local-nav-links .product-name {
            margin-right: auto;
            font-size: 3rem;
            font-weight: bold;
        }

        .local-nav-links a {
            font-size: 1rem;
            color: rgb(29, 29, 30);
            text-decoration: none;
        }

        .local-nav-links a:not(.product-name) {
            margin-left: 2em;
        }

        .first-image{
            padding-top: 30px;
            with: 
            overflow: hidden;
            margin: 0 auto;
        }
        
        .first-image-src{
            with: 100%;
            height: 100%;
            object-fill: cover;
        }

        .main-message p {
            font-weight: bold;
            text-align: center;
            line-height: 1.2;
        }

        .emoji1 {
            padding-top: 18vh;
            text-align: center;
            font-size: 10rem;
        }

        .emptybox1 {
            padding-top: 0vh;
        }
                
    </style>

    <div class = "local-nav-links">
        <a href = "#" class = "product-name">AI Service</a>
        <a href = "#">제품</a>
        <a href = "#">솔루션</a>
        <a href = "#">더보기</a>
    </div>

    <div class = "emptybox1"></div>

"""
st.markdown(html_css, unsafe_allow_html=True)

col1, col2 = st.columns(2)



def watsonx_ai_api(prompts):
    payload = {"prompt": prompts}
    response_data = requests.post(api_url, json = payload)
    response = response_data.json()
    print("generated_text:", response_data.json())
  
    return response['text']

# with st.sidebar:
#     api_url = st.text_input('Enter API Url:', value="http://localhost:8000/processing")
  
    if not (api_url):
        st.warning('Please enter your LLM Serving API Url!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = watsonx_ai_api(prompt) 
            st.write(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})

