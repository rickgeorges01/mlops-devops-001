# Importation des bibliothèques nécessaires
import mlflow
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

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration MLflow - CORRECTION MAJEURE !
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# Chemins utilisés par l'application
UPLOAD_PATH = "uploaded_image.jpg"

# Chargement depuis MLflow Registry
MODEL_NAME = "Fruit_Classification"

print(f" Connexion à MLflow : {MLFLOW_URI}")
print(f" Chargement du modèle : {MODEL_NAME}")

try:
    # Chargement du modèle depuis MLflow Registry
    model_uri = f"models:/{MODEL_NAME}"
    loaded_model = mlflow.keras.load_model(model_uri)
    print(" Modèle chargé avec succès depuis MLflow Registry")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    loaded_model = None


# Fonction de prétraitement : redimensionnement + normalisation de l'image
def preprocess_image(image_path):
    """Préprocess l'image pour qu'elle soit compatible avec le modèle"""
    # Redimension
    image = load_img(image_path, target_size=(224, 224))
    # Normalisation [0, 1]
    image_array = img_to_array(image) / 255.0
    # Conversion en batch (forme: (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# Page HTML basique pour uploader une image
@app.get("/")
def home():
    """Page d'accueil avec formulaire d'upload"""
    model_status = " Connecté" if loaded_model is not None else "Erreur"

    return HTMLResponse(f"""
    <html>
        <body>
            <h2> Classificateur de Fruits</h2>
            <p><strong>Status du modèle :</strong> {model_status}</p>
            <p><strong>MLflow URI :</strong> {MLFLOW_URI}</p>

            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value=" Prédire">
            </form>

            <hr>
            <p><em>Fruits supportés : Apple, Avocado, Banana, Cherry, Kiwi, Mango, Orange, Pineapple, Strawberry, Watermelon</em></p>
        </body>
    </html>
    """)


# Endpoint de prédiction : reçoit une image et retourne une étiquette
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint principal pour la prédiction d'images"""

    # Vérification du modèle
    if loaded_model is None:
        return JSONResponse(
            content={"error": "Modele indisponible. Verifiez la connexion MLflow."},
            status_code=500
        )

    # Vérifie si le fichier est une image supportée
    if not file.filename.lower().endswith(("jpg", "jpeg", "png", "bmp", "tiff")):
        return JSONResponse(
            content={"error": "Format non supporté. Utilisez : jpg, jpeg, png, bmp, tiff"},
            status_code=400
        )

    try:
        # Sauvegarde temporaire de l'image sur le disque
        with open(UPLOAD_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Dictionnaire d'étiquettes : classes des fruits
        label_dict = {
            'Apple': 0, 'Avocado': 1, 'Banana': 2, 'Cherry': 3, 'Kiwi': 4,
            'Mango': 5, 'Orange': 6, 'Pineapple': 7, 'Strawberry': 8, 'Watermelon': 9
        }
        index_to_label = {v: k for k, v in label_dict.items()}

        # Prétraitement de l'image + prédiction
        input_image = preprocess_image(UPLOAD_PATH)
        prediction = loaded_model.predict(input_image)

        # Classe avec la plus haute probabilité
        predicted_class = int(np.argmax(prediction))

        # Conversion en label texte
        predicted_label = index_to_label[predicted_class]

        # Niveau de confiance
        confidence = float(np.max(prediction))

        # Nettoyage : suppression du fichier temporaire
        if os.path.exists(UPLOAD_PATH):
            os.remove(UPLOAD_PATH)

        return JSONResponse(content={
            "label": predicted_label,
            "confidence": round(confidence * 100, 2),
            "filename": file.filename
        })

    except Exception as e:
        # Nettoyage en cas d'erreur
        if os.path.exists(UPLOAD_PATH):
            os.remove(UPLOAD_PATH)

        return JSONResponse(
            content={"error": f"Erreur lors de la prédiction : {str(e)}"},
            status_code=500
        )


# Endpoint pour visualiser l'image uploadée
@app.get("/image")
def get_image():
    """Affiche la dernière image uploadée (si elle existe encore)"""
    if os.path.exists(UPLOAD_PATH):
        return FileResponse(UPLOAD_PATH, media_type="image/jpeg")
    return HTMLResponse("<h3>Aucune image trouvée.</h3>", status_code=404)


# Endpoint de santé pour vérifier l'état du service
@app.get("/health")
def health_check():
    """Endpoint de monitoring pour vérifier l'état du service"""
    return JSONResponse(content={
        "status": "healthy" if loaded_model is not None else "unhealthy",
        "mlflow_uri": MLFLOW_URI,
        "model_loaded": loaded_model is not None
    })