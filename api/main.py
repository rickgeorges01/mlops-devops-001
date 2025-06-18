# Importation des bibliothèques nécessaires
import mlflow.keras  
from fastapi import FastAPI, UploadFile, File  
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse  
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from PIL import Image
import io
import os
import shutil

# Initialisation de l’application FastAPI
app = FastAPI()

# Chemins utilisés par l’application
UPLOAD_PATH = "uploaded_image.jpg"  
# Chemin du modèle MLflow (doit correspondre à celui utilisé dans train.py)
MODEL_PATH = "../mlflow/Fruit_Classification_model" 

# Chargement du modèle depuis MLflow (log_model utilisé dans train.py)
loaded_model = mlflow.keras.load_model(MODEL_PATH)


# Fonction de prétraitement : redimensionnement + normalisation de l’image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Redimension
    image_array = img_to_array(image) / 255.0  # Normalisation [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Conversion en batch (forme: (1, 224, 224, 3))
    return image_array

# Page HTML basique pour uploader une image (on vous conseilles les images depuis le dossier data/predict qui sont prevues a cet effet)
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

# Endpoint de prédiction : reçoit une image et retourne une étiquette
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérifie si le fichier est une image supportée
    if not file.filename.endswith(("jpg", "jpeg", "png")):
        return JSONResponse(content={"error": "Fichier non supporté"}, status_code=400)

    # Sauvegarde temporaire de l’image sur le disque
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Dictionnaire d’étiquettes : classes des fruits
    label_dict = {'Apple': 0, 'Avocado': 1, 'Banana': 2, 'Cherry': 3, 'Kiwi': 4, 'Mango': 5,
                  'Orange': 6, 'Pineapple': 7, 'Strawberry': 8, 'Watermelon': 9}
    index_to_label = {v: k for k, v in label_dict.items()}

    try:
        # Prétraitement de l’image + prédiction
        input_image = preprocess_image(UPLOAD_PATH)
        prediction = loaded_model.predict(input_image)

        predicted_class = int(np.argmax(prediction))  # Classe avec la plus haute probabilité
        predicted_label = index_to_label[predicted_class]  # Conversion en label texte

        return JSONResponse(content={"label": predicted_label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint pour visualiser l’image uploadée
@app.get("/image")
def get_image():
    if os.path.exists(UPLOAD_PATH):
        return FileResponse(UPLOAD_PATH, media_type="image/jpeg")
    return HTMLResponse("<h3>Aucune image trouvée.</h3>", status_code=404)
