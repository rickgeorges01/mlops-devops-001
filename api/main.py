# Importation des bibliothèques nécessaires
import mlflow
import mlflow.keras
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import os
import shutil
from pathlib import Path

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# Chemins utilisés par l'application
UPLOAD_PATH = "uploaded_image.jpg"

# Variables globales
MODEL_NAME = "Fruit_Classification_model"
MODEL_VERSION = "1"
loaded_model = None


def get_model():
    """Charge le modèle à la demande"""
    global loaded_model

    if loaded_model is None:
        try:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            loaded_model = mlflow.keras.load_model(model_uri)
            print("Modèle chargé avec succès depuis MLflow Registry")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return None

    return loaded_model

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

def get_test_images():
    """Scan le dossier test et retourne la liste des images disponibles"""
    test_images = []
    test_data_path = Path("/app/data/test")

    if not test_data_path.exists():
        print(f"Dossier test non trouvé : {test_data_path}")
        return []

    # Parcourir chaque dossier de fruit
    for fruit_folder in test_data_path.iterdir():
        if fruit_folder.is_dir():
            fruit_name = fruit_folder.name

            # Parcourir les images dans chaque dossier
            for image_file in fruit_folder.iterdir():
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Chemin relatif depuis /app/data/test
                    relative_path = f"{fruit_name}/{image_file.name}"
                    test_images.append({
                        "path": relative_path,
                        "display_name": f"{fruit_name} - {image_file.name}",
                        "fruit": fruit_name
                    })

    return sorted(test_images, key=lambda x: x["fruit"])

# Page HTML basique pour uploader une image
@app.get("/")
def home():
    """Page d'accueil avec double option"""
    model_status = "Connecté" if get_model() is not None else "Erreur"

    # Récupérer la liste des images test
    test_images = get_test_images()

    # Créer les options pour le dropdown
    image_options = ""
    for img in test_images:
        image_options += f'<option value="{img["path"]}">{img["display_name"]}</option>\n'

    return HTMLResponse(f"""
    <html>
        <body>
            <h2>Classificateur de Fruits</h2>
            <p><strong>Status du modèle :</strong> {model_status}</p>
            <p><strong>MLflow URI :</strong> {MLFLOW_URI}</p>

            <!-- Option 1: Images du dataset test -->
            <h3>Option 1: Tester avec images de référence</h3>
            <form action="/predict-from-dataset" method="post">
                <select name="test_image" required>
                    <option value="">Choisir une image...</option>
                    {image_options}
                </select>
                <input type="submit" value="Prédire">
            </form>

            <hr>

            <!-- Option 2: Upload personnel -->
            <h3>Option 2: Uploader votre image</h3>
            <form action="/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*" required>
                <input type="submit" value="Prédire">
            </form>

            <hr>
            <p><em>Fruits supportés : Apple, Avocado, Banana, Cherry, Kiwi, Mango, Orange, Pineapple, Strawberry, Watermelon</em></p>
        </body>
    </html>
    """)


# Endpoint de prédiction : reçoit une image de l'utilisateur et retourne une étiquette
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint principal pour la prédiction d'images"""

    # Chargement à la demande
    model = get_model()
    if model is None:
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
        prediction = model.predict(input_image)

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
            "filename": file.filename,
            "source": "upload_by_user"
        })

    except Exception as e:
        # Nettoyage en cas d'erreur
        if os.path.exists(UPLOAD_PATH):
            os.remove(UPLOAD_PATH)

        return JSONResponse(
            content={"error": f"Erreur lors de la prédiction : {str(e)}"},
            status_code=500
        )

# Endpoint de prédiction :  une image sorti du dataset fourni et retourne une étiquette
@app.post("/predict-from-dataset")
async def predict_from_dataset(test_image: str = Form(...)):
    """Endpoint pour prédiction avec images du dataset test"""

    # Vérification du modèle
    model = get_model()
    if model is None:
        return JSONResponse(
            content={"error": "Modele indisponible. Verifiez la connexion MLflow."},
            status_code=500
        )

    # Vérification que l'image est sélectionnée
    if not test_image:
        return JSONResponse(
            content={"error": "Aucune image sélectionnée"},
            status_code=400
        )

    try:
        # Construction du chemin complet
        image_path = f"/app/data/test/{test_image}"

        # Vérification que le fichier existe
        if not os.path.exists(image_path):
            return JSONResponse(
                content={"error": f"Image non trouvée : {test_image}"},
                status_code=404
            )

        # Dictionnaire d'étiquettes (même que dans predict)
        label_dict = {
            'Apple': 0, 'Avocado': 1, 'Banana': 2, 'Cherry': 3, 'Kiwi': 4,
            'Mango': 5, 'Orange': 6, 'Pineapple': 7, 'Strawberry': 8, 'Watermelon': 9
        }
        index_to_label = {v: k for k, v in label_dict.items()}

        # Prétraitement et prédiction (même logique)
        input_image = preprocess_image(image_path)
        prediction = model.predict(input_image)

        predicted_class = int(np.argmax(prediction))
        predicted_label = index_to_label[predicted_class]
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "label": predicted_label,
            "confidence": round(confidence * 100, 2),
            "filename": test_image,
            "source": "dataset_test"
        })

    except Exception as e:
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
        "status": "healthy" if get_model() is not None else "unhealthy",
        "mlflow_uri": MLFLOW_URI,
        "model_loaded": get_model() is not None
    })


