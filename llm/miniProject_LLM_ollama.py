#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# Ollama 서버 설정
OLLAMA_SERVER = "http://localhost:11434"
MODEL_NAME = "gemma2"
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_SERVER, temperature=0.1)


# In[ ]:


# ConversationalRetrievalChain 설정
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=False
)


# In[ ]:


# TensorFlow MobileNetV2 모델 로드
model = tf.keras.applications.MobileNetV2(weights="imagenet")


# In[ ]:


# 음식 정보 추출 함수 (재료, 영양성분)
def get_food_info_from_vectorstore(food_name, vectorstore):
    try:
        # food_name을 쿼리로 변환
        query = f"Tell me about {food_name}"

        # 쿼리를 벡터화
        query_vector = vectorstore.embeddings.embed_query(query)

        # FAISS에서 검색
        D, I = vectorstore.index.search(np.array([query_vector]).astype('float32'), 1)  # 상위 1개 검색

        # 가장 유사한 음식 정보 가져오기
        if I[0][0] != -1:  # 유효한 결과가 있을 경우
            # vectorstore에 있는 음식 정보 가져오기 (food_name을 통해)
            matched_food = vectorstore.docs[I[0][0]]
            return matched_food
        else:
            return {"error": "음식 정보를 찾을 수 없습니다."}
    except Exception as e:
        return {"error": f"Error: {e}"}


# In[ ]:


# Ollama를 사용해 음식 설명 생성
def get_food_description_with_langchain(food_name):
    """
    LangChain ChatOllama를 사용하여 음식 설명 생성
    """
    try:
        response = llm([HumanMessage(content=f"Tell me about {food_name}.")])
        if isinstance(response, AIMessage):
            return response.content
        return f"Unexpected response: {response}"
    except Exception as e:
        return f"Failed to retrieve description: {e}"


# In[ ]:


# 음식 재료와 영양 정보 찾기
def get_food_nutrition_and_ingredient(predicted_ingredient):
    # Step 1: 영양소 정보 가져오기
    matching_nutrition_row = concatenated_food_nutrition[concatenated_food_nutrition['food'] == predicted_ingredient]
    
    if not matching_nutrition_row.empty:
        nutrition_info = matching_nutrition_row.iloc[0].to_dict()  # 첫 번째 행을 딕셔너리로 변환
    else:
        nutrition_info = {"error": "No matching food found in nutrition data"}
    
    # Step 2: 재료 정보 가져오기
    matching_ingredient_row = food_ingredient[food_ingredient['Title'] == predicted_ingredient]
    
    if not matching_ingredient_row.empty:
        ingredient_info = matching_ingredient_row.iloc[0].to_dict()  # 첫 번째 행을 딕셔너리로 변환
    else:
        ingredient_info = {"error": "No matching ingredient found"}
    
    return nutrition_info, ingredient_info


# In[ ]:


def filter_dataframes_by_food_name(food_name, concatenated_food_nutrition, food_ingredient):
    """
    특정 food_name이 포함된 데이터를 각 데이터프레임에서 필터링하고,
    Unnamed로 시작하는 컬럼은 제외합니다.
    """
    # "Unnamed" 컬럼 제거
    nutrition_filtered = concatenated_food_nutrition.loc[
        concatenated_food_nutrition["food"].str.contains(food_name, case=False, na=False)
    ]
    nutrition_filtered = nutrition_filtered.loc[:, ~nutrition_filtered.columns.str.startswith("Unnamed")]

    ingredient_filtered = food_ingredient.loc[
        food_ingredient["Title"].str.contains(food_name, case=False, na=False)
    ]
    ingredient_filtered = ingredient_filtered.loc[:, ~ingredient_filtered.columns.str.startswith("Unnamed")]

    return nutrition_filtered, ingredient_filtered


# In[ ]:


# 빈 데이터 프레임 대체 문구
def handle_empty_dataframe(df, no_info_message="No info"):
    """
    빈 데이터프레임을 no_info_message로 채워진 데이터프레임으로 변환
    """
    if df.empty:
        columns = ["Column1"] if df.columns.empty else df.columns  # 기본 컬럼 이름 생성
        return pd.DataFrame({col: [no_info_message] for col in columns})
    return df


# In[ ]:


# 이미지 예측 함수

def predict_image_with_description(image_url):
    """
    이미지 URL을 받아 음식 예측과 Ollama 설명, 데이터 필터링 결과를 반환
    """
    try:
        # 이미지 처리
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).resize((224, 224))

        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.expand_dims(image_array, axis=0)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

        # 이미지 예측
        predictions = model.predict(image_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        result = {label: float(prob) for (_, label, prob) in decoded_predictions}

        # 가장 높은 확률의 예측값
        top_food = decoded_predictions[0][1]

        # Ollama 설명 생성
        description = get_food_description_with_langchain(top_food)

        # 데이터프레임 필터링
        nutrition_filtered, ingredient_filtered = filter_dataframes_by_food_name(
            top_food, concatenated_food_nutrition, food_ingredient
        )

        # 빈 데이터프레임 처리 (영양 정보와 재료 정보에 각각 처리)
        nutrition_output = handle_empty_dataframe(nutrition_filtered)
        ingredient_output = handle_empty_dataframe(ingredient_filtered)

        # 결과 반환 (표 형식으로 변환)
        return result, description, nutrition_output, ingredient_output
    
    except Exception as e:
        return {"error": 1.0}, f"Error: {e}", "No info", "No info"

