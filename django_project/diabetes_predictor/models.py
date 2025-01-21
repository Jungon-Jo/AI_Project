from django.db import models

class PredictionResult(models.Model):
    age = models.FloatField()
    bmi = models.FloatField()
    glucose = models.FloatField()
    blood_pressure = models.FloatField()
    insulin = models.FloatField()
    diabetes_pedigree = models.FloatField()
    pregnancies = models.IntegerField()
    skin_thickness = models.FloatField()
    prediction = models.BooleanField()
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for Age: {self.age}, BMI: {self.bmi}"