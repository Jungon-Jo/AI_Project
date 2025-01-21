#!/usr/bin/env python
# coding: utf-8

# In[203]:


import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from langchain_community.chat_models import ChatOllama


# In[ ]:


# Gradio 인터페이스 설정
with gr.Blocks() as iface:
    # 제목 및 설명
    gr.Markdown("# 음식 재료 및 영양 정보")
    gr.Markdown("이미지 URL을 입력하면 해당 음식에 대한 정보를 출력합니다.")

    # 입력 필드, 출력 필드 (예측 결과)
    with gr.Row():
        with gr.Column():
            image_url_input = gr.Textbox(label="이미지 URL 입력")
        with gr.Column():
            prediction_label = gr.Label(num_top_classes=3, label="예측 결과")
    # 출력 필드 (설명 및 표)
    with gr.Row():
        food_description_output = gr.Textbox(label="음식 설명", interactive=False)
    
    with gr.Row():
        nutrition_info_output = gr.DataFrame(label="영양 정보", interactive=False)
    
    with gr.Row():
        ingredient_info_output = gr.DataFrame(label="재료 정보", interactive=False)

    # 버튼으로 실행
    predict_button = gr.Button("예측 실행")
    predict_button.click(
        fn=predict_image_with_description,
        inputs=[image_url_input],
        outputs=[prediction_label, food_description_output, nutrition_info_output, ingredient_info_output],
    )


# In[ ]:


# 인터페이스 실행
iface.launch(server_port=7861, server_name="0.0.0.0", debug=True)


# In[337]:


iface.close()

