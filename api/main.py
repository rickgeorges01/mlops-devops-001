import mlflow.keras
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import os
import shutil

app = FastAPI()

UPLOAD_PATH = "uploaded_image.jpg"
MODEL_PATH = "../mlflow/Fruit_Classification_model"  # chemin où train.py a sauvegardé le modèle
loaded_model = mlflow.keras.load_model(MODEL_PATH)

# (Optionnel) Stocker manuellement une métrique accuracy ici si vous ne la récupérez pas dynamiquement
model_accuracy = 0.92  # à remplacer par un chargement réel ou via MLflow Tracking si nécessaire

# Fonction pour prétraiter l'image avant la prédiction
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Route pour afficher la page d'accueil avec le formulaire de téléchargement
@app.get("/")
def home():
    return HTMLResponse("""
    <html>
        <body>
            <h2>Uploader une image</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Prédire">
            </form>
        </body>
    </html>
    """)

# Route pour prédire la classe de l'image téléchargée
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(("jpg", "jpeg", "png")):
        return JSONResponse(content={"error": "Fichier non supporté"}, status_code=400)

    # Sauvegarder l'image
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    label_dict = {'Apple': 0, 'Avocado': 1,'Banana': 2,'Cherry':3,'Kiwi':4,'Mango':5,'Orange':6,'Pineapple':7,'Strawberry':8,'Watermelon':9}
    index_to_label = {v: k for k, v in label_dict.items()}

    try:
     input_image = preprocess_image(UPLOAD_PATH)
     prediction = loaded_model.predict(input_image)
     predicted_class = int(np.argmax(prediction))  # Index prédite

     predicted_label = index_to_label[predicted_class]

     return JSONResponse(content={
         "label": predicted_label
     })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Route pour afficher l'image téléchargée
@app.get("/image")
def get_image():
    if os.path.exists(UPLOAD_PATH):
        return FileResponse(UPLOAD_PATH, media_type="image/jpeg")
    return HTMLResponse("<h3>Aucune image trouvée.</h3>", status_code=404)
