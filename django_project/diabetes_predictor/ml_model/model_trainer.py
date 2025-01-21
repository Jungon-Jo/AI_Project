import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class DiabetesModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train_model(self):
        # 현재 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 프로젝트 루트 디렉토리 경로 계산
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # 데이터셋 파일 경로
        dataset_path = os.path.join(project_root, 'django_project/dataset', 'diabetes.csv')
        
        # 데이터 로드
        try:
            data = pd.read_csv(dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다. 경로: {dataset_path}")
        
        # 특성과 타겟 분리
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # 데이터 전처리
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 생성 및 학습
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # 모델 저장 경로
        model_save_path = os.path.join(current_dir, 'diabetes_model.joblib')
        scaler_save_path = os.path.join(current_dir, 'scaler.joblib')
        
        # 모델 저장
        joblib.dump(self.model, model_save_path)
        joblib.dump(self.scaler, scaler_save_path)
        
        print("모델 학습 및 저장이 완료되었습니다.")
        print(f"모델 저장 경로: {model_save_path}")
        print(f"스케일러 저장 경로: {scaler_save_path}")

if __name__ == "__main__":
    trainer = DiabetesModelTrainer()
    trainer.train_model()