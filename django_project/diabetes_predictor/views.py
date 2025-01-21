from django.shortcuts import render
from .forms import DiabetesPredictionForm
import joblib
import numpy as np
from .models import PredictionResult
import os

def get_model_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'ml_model', 'diabetes_model.joblib')
    scaler_path = os.path.join(current_dir, 'ml_model', 'scaler.joblib')
    return model_path, scaler_path

def home(request):
    if request.method == 'POST':
        form = DiabetesPredictionForm(request.POST)
        if form.is_valid():
            # 모델과 스케일러 로드
            model_path, scaler_path = get_model_path()
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # 입력 데이터 준비
            features = np.array([[
                form.cleaned_data['pregnancies'],
                form.cleaned_data['glucose'],
                form.cleaned_data['blood_pressure'],
                form.cleaned_data['skin_thickness'],
                form.cleaned_data['insulin'],
                form.cleaned_data['bmi'],
                form.cleaned_data['diabetes_pedigree'],
                form.cleaned_data['age']
            ]])
            
            # 데이터 스케일링 및 예측
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
            
            # 결과 저장
            result = PredictionResult(
                pregnancies=form.cleaned_data['pregnancies'],
                glucose=form.cleaned_data['glucose'],
                blood_pressure=form.cleaned_data['blood_pressure'],
                skin_thickness=form.cleaned_data['skin_thickness'],
                insulin=form.cleaned_data['insulin'],
                bmi=form.cleaned_data['bmi'],
                diabetes_pedigree=form.cleaned_data['diabetes_pedigree'],
                age=form.cleaned_data['age'],
                prediction=prediction,
                probability=probability
            )
            result.save()
            
            return render(request, 'diabetes_predictor/result.html', {
                'prediction': prediction,
                'probability': probability * 100
            })
    else:
        form = DiabetesPredictionForm()
    
    return render(request, 'diabetes_predictor/index.html', {'form': form})