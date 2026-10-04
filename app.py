
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Charger le modèle entraîné
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Récupérer les données du formulaire
            brand = request.form["brand"]
            ram = float(request.form["ram"])
            battery = float(request.form["battery"])
            camera = float(request.form["camera"])
            screen_size = float(request.form["screen_size"])
            launch_year = int(request.form["launch_year"])

            # Créer un DataFrame avec les mêmes colonnes que le notebook
            smartphone = pd.DataFrame([{
                "brand": brand,
                "ram": ram,
                "battery": battery,
                "camera": camera,
                "screen_size": screen_size,
                "launch_year": launch_year
            }])

            # Faire la prédiction
            prix = model.predict(smartphone)[0]

            prediction = round(float(prix), 2)

        except ValueError:
            error = "Veuillez saisir des valeurs numériques valides."
        except Exception as e:
            error = f"Une erreur est survenue : {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
    