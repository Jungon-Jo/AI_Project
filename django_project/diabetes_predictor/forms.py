from django import forms

class DiabetesPredictionForm(forms.Form):
    pregnancies = forms.IntegerField(label='임신 횟수', min_value=0)
    glucose = forms.FloatField(label='포도당 수치', min_value=0)
    blood_pressure = forms.FloatField(label='혈압', min_value=0)
    skin_thickness = forms.FloatField(label='피부 두께', min_value=0)
    insulin = forms.FloatField(label='인슐린', min_value=0)
    bmi = forms.FloatField(label='BMI', min_value=0)
    diabetes_pedigree = forms.FloatField(label='당뇨병 가족력', min_value=0)
    age = forms.FloatField(label='나이', min_value=0)