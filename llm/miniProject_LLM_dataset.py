#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


import os
print(os.getcwd())  # 현재 작업 디렉토리 확인

# In[ ]:


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


# In[ ]:


# 텍스트 분할
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts_nutrition = text_splitter.split_text("\n".join(concatenated_food_nutrition.to_string()))
texts_ingredient = text_splitter.split_text("\n".join(food_ingredient.to_string()))


# In[ ]:


# 텍스트를 Document 객체로 변환 (FAISS가 요구하는 형식)
documents_nutrition = [Document(page_content=text) for text in texts_nutrition]
documents_ingredient = [Document(page_content=text) for text in texts_ingredient]

# 임베딩 초기화
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 각 텍스트에 대해 임베딩 생성
embeddings_nutrition = embedding_model.embed_documents(texts_nutrition)
embeddings_ingredient = embedding_model.embed_documents(texts_ingredient)


# In[ ]:


# 벡터 데이터베이스 생성
all_dataset = documents_nutrition + documents_ingredient
# all_embedding = embeddings_nutrition_vectors + embeddings_ingredient_vectors
vectorstore = FAISS.from_documents(all_dataset, embedding_model)


# In[ ]:


# 벡터 데이터베이스 저장
# vectorstore.save_local("../llm")
print(type(vectorstore))
