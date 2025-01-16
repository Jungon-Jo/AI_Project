#!/usr/bin/env python
# coding: utf-8

# 

# In[228]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix # "classification_report" import 불가 
import streamlit as st


# In[229]:


# 폰트지정
plt.rcParams["font.family"] = "NanumGothic"

# 마이너스 부호 깨짐 지정
plt.rcParams["axes.unicode_minus"] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = "{:.2f}".format


# In[230]:


# 데이터 로드
file_path1 = "../dataset_miniProject/miniProject_coldcode_city.csv" # 데이터 파일 경로
file_path2 = "../dataset_miniProject/humidity_data.csv"
file_path3 = "../dataset_miniProject/rain_data.csv"
file_path4 = "../dataset_miniProject/temperature_data.csv"
df1 = pd.read_csv(file_path1, encoding="cp949")
df2 = pd.read_csv(file_path2, encoding="cp949")
df3 = pd.read_csv(file_path3, encoding="cp949")
df4 = pd.read_csv(file_path4, encoding="cp949")


# In[231]:


# 기본 데이터 정보 확인
print("데이터 크기:", df1.shape)
print("데이터 크기:", df2.shape)
print("데이터 크기:", df3.shape)
print("데이터 크기:", df4.shape)
print("\n처음 5개 행(진료데이터):")
# display(df1.head())
print("\n처음 5개 행(습도):")
# display(df2.head())
print("\n처음 5개 행(강수):")
# display(df3.head())
print("\n처음 5개 행(기온):")
# display(df4.head())
print("\n데이터 기간:", df1["날짜"].min(), "~", df1["날짜"].max())
print("총 관측 수:", len(df1))
print("\n데이터 기간:", df2["날짜"].min(), "~", df2["날짜"].max())
print("총 관측 수:", len(df2))
print("\n데이터 기간:", df3["날짜"].min(), "~", df3["날짜"].max())
print("총 관측 수:", len(df3))
print("\n데이터 기간:", df4["날짜"].min(), "~", df4["날짜"].max())
print("총 관측 수:", len(df4))


# In[232]:


# 데이터 전처리

# 진료건수
# 날짜 형식 변경
# 날짜를 0000-00-00 형태의 datetime으로 변경
# 일 단위 삭제(문자열로 변경) 후, 다시 datetime으로 변경(일이 01일로 통일)
df1["날짜"] = df1["날짜"].str.replace(r"[.-]", "-", regex=True)
df1["날짜"] = pd.to_datetime(df1["날짜"], format="%Y-%m-%d")
df1["날짜"] = df1["날짜"].dt.strftime('%Y-%m')
df1["날짜"] = pd.to_datetime(df1["날짜"], format="%Y-%m")
# print(df1["날짜"])
# print(df1["날짜"].dtype)

# 날짜 기준 설정 및 적용
min_date1 = pd.to_datetime("2014-12")
max_date1 = pd.to_datetime("2023-12")
coldConsultation_filteredByDate = df1[(df1["날짜"] > min_date1) & (df1["날짜"] <= max_date1)]

# 결측치 처리
# 결측치를 평균 값으로 채우겠다
coldConsultation_filteredByDate["발생건수(건)"] = coldConsultation_filteredByDate["발생건수(건)"].fillna(coldConsultation_filteredByDate["발생건수(건)"].mean())

# 지역코드 11(서울)만 추출
coldConsultation_seoul = coldConsultation_filteredByDate[coldConsultation_filteredByDate["시도지역코드"] == 11]
print("\n지역코드 추출 데이터 set:")
print(coldConsultation_seoul)

# 동일 날짜의 모든지역 진료건수 더하기(월별 데이터로 변환)
coldConsultation_seoul = coldConsultation_seoul.groupby("날짜")["발생건수(건)"].sum().reset_index()
print("\n동일 날짜 기준 데이터 set:")
print(coldConsultation_seoul)

# 날씨
# 날짜 형식 변경 함수
# 연도별로 달리 처리: 2015년은 15로, 그 외는 연도의 끝 2자리를 사용
def convert_date_format(date_str):
    month, year_end = date_str.split("-")
    
    # 2015년인 경우
    if year_end == "15":
        return f"2015-{str(month).zfill(2)}"
    
    # 2016년 이후는 연도 끝 두 자리를 이용해 "2016-01", "2017-02" 형태로 변환
    return f"20{year_end}-{str(month).zfill(2)}"

# 날짜 형식 변경
df2["날짜"] = df2["날짜"].apply(convert_date_format)
df3["날짜"] = df3["날짜"].apply(convert_date_format)
df4["날짜"] = df4["날짜"].apply(convert_date_format)
df2["날짜"] = pd.to_datetime(df2["날짜"], format="%Y-%m")
df3["날짜"] = pd.to_datetime(df3["날짜"], format="%Y-%m")
df4["날짜"] = pd.to_datetime(df4["날짜"], format="%Y-%m")

# 날짜 기준 설정 및 적용
df2["날짜"] = pd.to_datetime(df2["날짜"], format="%Y-%m")
df3["날짜"] = pd.to_datetime(df3["날짜"], format="%Y-%m")
df4["날짜"] = pd.to_datetime(df4["날짜"], format="%Y-%m")
min_date = pd.to_datetime("2014-12")
max_date = pd.to_datetime("2023-12")
seoulHumidity_filteredByDate = df2[(df2["날짜"] > min_date) & (df2["날짜"] <= max_date)]
seoulRain_filteredByDate = df3[(df3["날짜"] > min_date) & (df3["날짜"] <= max_date)]
seoulTemperature_filteredByDate = df4[(df4["날짜"] > min_date) & (df4["날짜"] <= max_date)]
print("\n습도 데이터 set:")
print(seoulHumidity_filteredByDate)
print("\n강수 데이터 set:")
print(seoulRain_filteredByDate)
print("\n기온 데이터 set:")
print(seoulTemperature_filteredByDate)

# 날짜를 기준으로 두 데이터값 merge
data_merge_humidity = pd.merge(coldConsultation_seoul, seoulHumidity_filteredByDate, on="날짜", how="inner")
data_merge_rain = pd.merge(data_merge_humidity, seoulRain_filteredByDate, on="날짜", how="inner")
data_merge_total = pd.merge(data_merge_rain, seoulTemperature_filteredByDate, on="날짜", how="inner")
print("\n최종 데이터 set:")
print(data_merge_total)

# NaN값을 평균값으로 대체
data_merge_total["1시간최다강수량(mm)"] = data_merge_total["1시간최다강수량(mm)"].fillna(data_merge_total["1시간최다강수량(mm)"].mean())

# 날짜를 숫자형으로 변환
data_merge_total['날짜(int)'] = data_merge_total['날짜'].astype(int) / 10**9  # Unix 타임스탬프로 변환 (초 단위)


# In[233]:


# 전처리된 데이터 확인
print("\n전처리된 데이터 샘플:")
# display(data_merge_total.head())


# In[234]:


data_merge_total["year"] = data_merge_total["날짜"].dt.year
data_merge_total["month"] = data_merge_total["날짜"].dt.month
# display(data_merge_total.head())


# In[235]:


# 감기 진료 건수 모델 학습
features_coldConsultation = ["평균습도(%rh)", "강수량(mm)", "평균기온(℃)", "평균 일교차"]
X_coldConsultation = data_merge_total[features_coldConsultation]
y_coldConsultation = data_merge_total["발생건수(건)"]
print(X_coldConsultation)
print(y_coldConsultation)


# In[236]:


# 데이터 분할
X_train_coldConsultation, X_test_coldConsultation, y_train_coldConsultation, y_test_coldConsultation = train_test_split(X_coldConsultation, y_coldConsultation, test_size=0.2, random_state=42)
# 데이터를 표준화합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_coldConsultation)
X_test_scaled = scaler.transform(X_test_coldConsultation)


# In[237]:


# 모델 학습
model_coldConsultation = LinearRegression()
model_coldConsultation.fit(X_train_scaled, y_train_coldConsultation)


# In[238]:


model_coldConsultation.score(X_test_scaled, y_test_coldConsultation)
print(model_coldConsultation.score(X_test_scaled, y_test_coldConsultation))


# In[239]:


# 모델 예측
y_pred_coldConsultation = model_coldConsultation.predict(X_test_scaled)
print(y_pred_coldConsultation)


# In[240]:


# 모델 평가
print("\ncoldConsultation 모델 성능")
print(f"R2 Score: {r2_score(y_test_coldConsultation, y_pred_coldConsultation):.4f}")
print(f"MSE: {mean_squared_error(y_test_coldConsultation, y_pred_coldConsultation):.4f}")


# In[241]:


humidity = st.number_input("습도(%) : ", min_value=0, max_value=100, step=5)
rain = st.number_input("강수량(mm) : ", min_value=0, max_value=500, step=10)
temperature = st.number_input("기온(℃) : ", min_value=-30, max_value=50, step=1)
temperature_range = st.number_input("일교차(℃) : ", min_value=0, max_value=40, step=1)

future_date = pd.DataFrame({
    "평균습도(%rh)": [humidity],        # 예: 2025년 3월 예상 평균 습도
    "강수량(mm)": [rain],        # 예: 2025년 3월 예상 강수량
    "평균기온(℃)": [temperature],       # 예: 2025년 3월 예상 평균 기온
    "평균 일교차": [temperature_range],        # 예: 2025년 3월 예상 평균 일교차
})

# 모델 예측
if st.button("예측 감기 발병 건수"):
    future_date_scaled = scaler.transform(future_date)
    predicted_coldConsultation = model_coldConsultation.predict(future_date_scaled)
    st.write(f"날씨에 따른 감기 발병 건수(예측): {round(predicted_coldConsultation[0], 2)}")

