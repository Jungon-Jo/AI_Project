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
import faiss
import pickle
from langchain.docstore import InMemoryDocstore


# # In[204]:


# CSV 파일 로드(재료 영양소)
food_nutrition1 = pd.read_csv("/Users/jojungon/AI_Project/dataset/FINAL_FOOD_DATASET/FOOD-DATA-GROUP1.csv", encoding='CP949')
food_nutrition2 = pd.read_csv("/Users/jojungon/AI_Project/dataset/FINAL_FOOD_DATASET/FOOD-DATA-GROUP2.csv", encoding='CP949')
food_nutrition3 = pd.read_csv("/Users/jojungon/AI_Project/dataset/FINAL_FOOD_DATASET/FOOD-DATA-GROUP3.csv", encoding='CP949')
food_nutrition4 = pd.read_csv("/Users/jojungon/AI_Project/dataset/FINAL_FOOD_DATASET/FOOD-DATA-GROUP4.csv", encoding='CP949')
food_nutrition5 = pd.read_csv("/Users/jojungon/AI_Project/dataset/FINAL_FOOD_DATASET/FOOD-DATA-GROUP5.csv", encoding='CP949')
concatenated_food_nutrition = pd.concat([food_nutrition1, food_nutrition2, food_nutrition3, food_nutrition4, food_nutrition5], ignore_index=True)
concatenated_food_nutrition.tail()


# In[ ]:


# CSV 파일 로드(음식 재료)
food_ingredient = pd.read_csv("/Users/jojungon/AI_Project/dataset/Food_Ingredients_Dataset.csv")
food_ingredient.tail()


# In[306]:


# 로컬 모델 파일 경로
FAISS_INDEX_PATH = "/Users/jojungon/AI_Project/llm/index.faiss"
PKL_METADATA_PATH = "/Users/jojungon/AI_Project/llm/index.pkl"

# FAISS 인덱스 불러오기
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# 메타데이터 불러오기
with open(PKL_METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# 임베딩 초기화
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 문서 저장소와 매핑 생성
docstore = InMemoryDocstore(metadata)
index_to_docstore_id = {i: str(i) for i in range(len(metadata))}

# 기존 FAISS 인덱스를 LangChain의 FAISS 객체로 래핑
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Ollama 서버 설정
OLLAMA_SERVER = "http://localhost:11434"
MODEL_NAME = "gemma2"
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_SERVER, temperature=0.1)


# In[307]:


# ConversationalRetrievalChain 설정
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
    verbose=False
)


# In[308]:


# TensorFlow MobileNetV2 모델 로드
model = tf.keras.applications.MobileNetV2(weights="imagenet")


# In[309]:


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


# In[310]:


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


# In[311]:


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


# In[312]:


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


# In[325]:


# 빈 데이터 프레임 대체 문구
def handle_empty_dataframe(df):
    """
    빈 데이터프레임을 그대로 반환하거나, 특정 컬럼 구조만 유지하는 빈 데이터프레임 반환
    """
    if df.empty:
        return pd.DataFrame(columns=["Info"])  # 기본 컬럼 구조를 반환
    return df




# In[332]:


# 이미지 예측 함수
def predict_image_with_description(image_url):
    """
    이미지 URL을 받아 음식 예측과 Ollama 설명, 데이터 필터링 결과를 반환
    """
    try:
        # 이미지 처리
        response = requests.get(image_url)
        if response.status_code != 200:
            return (
                {"error": 1.0},
                f"Error: Unable to fetch image from URL. Status code: {response.status_code}",
                pd.DataFrame(columns=["Info"]),
                pd.DataFrame(columns=["Info"]),
            )

        try:
            image = Image.open(BytesIO(response.content)).resize((224, 224))
        except Exception as e:
            return (
                {"error": 1.0},
                f"Error: Unable to process image. {e}",
                pd.DataFrame(columns=["Info"]),
                pd.DataFrame(columns=["Info"]),
            )

        # 이미지 데이터를 모델에 전달
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

        # 빈 데이터프레임 처리
        nutrition_output = handle_empty_dataframe(nutrition_filtered)
        ingredient_output = handle_empty_dataframe(ingredient_filtered)

        # 결과 반환
        return result, description, nutrition_output, ingredient_output

    except Exception as e:
        # 에러 발생 시 반환 값
        return (
            {"error": 1.0},
            f"Error: {e}",
            pd.DataFrame(columns=["Info"]),
            pd.DataFrame(columns=["Info"]),
        )



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
iface.launch(server_port=7861, server_name="0.0.0.0", debug=True, share=True)


# In[337]:


iface.close()


# In[ ]:




