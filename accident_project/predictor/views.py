from django.shortcuts import render
import joblib
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "..", "model", "accident_model.pkl")

model = joblib.load(model_path)


def home(request):
    return render(request, 'home.html')


def predict(request):

    if request.method == "POST":

        weather = int(request.POST.get("weather"))
        road = int(request.POST.get("road"))
        light = int(request.POST.get("light"))
        speed = int(request.POST.get("speed"))
        vehicles = int(request.POST.get("vehicles"))
        age = int(request.POST.get("age"))
        gender = int(request.POST.get("gender"))

        casualty = 2

        features = [[
            weather,
            road,
            light,
            speed,
            vehicles,
            age,
            gender,
            casualty
        ]]

        prediction = model.predict(features)

        severity_map = {
            1: "Severe",     # Fatal
            2: "Moderate",   # Serious
            3: "Minor"       # Slight
        }

        result = severity_map[prediction[0]]

        return render(request, "result.html", {"result": result})

    return render(request, "predict.html")