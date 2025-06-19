#!/usr/bin/env python

# About Author
# Project:` Fruit Classification Multiclass Problem
# Introduction
# 👋 Hello, everyone! My name is Mehak Iftikhar, and I'm delighted to introduce myself to you. I am a Junior Data Scientist passionate about leveraging data to derive meaningful insights and drive impactful decisions. With a keen interest in exploring the realms of data science, I actively engage in various projects and share my learnings through platforms like Kaggle.
# 
# About Me
# 🔍 As a Junior Data Scientist, I immerse myself in the world of data, constantly seeking innovative ways to analyze, interpret, and visualize information to solve real-world problems. My journey in data science is fueled by a curiosity to unravel patterns, discover trends, and uncover hidden insights within complex datasets
# 
# My Work
# 📊 I regularly upload my data analysis notebooks and projects on Kaggle, where I showcase my skills in data manipulation, exploratory data analysis (EDA), machine learning, and more. Through these notebooks, I aim to contribute to the data science community by sharing methodologies, code snippets, and insights gained from my projects.
# 
# Passion & Goals
# 💡 My passion for data science extends beyond technical skills. I am dedicated to continuous learning and improvement, staying updated with the latest advancements in the field. My ultimate goal is to harness the power of data to make a positive impact on society, whether it's through enhancing business strategies, addressing societal challenges, or driving innovation in various domains.
# 
# Let's Connect
# 🤝 I am always open to collaboration, knowledge sharing, and networking opportunities. Feel free to connect with me on Kaggle or other professional platforms to discuss data science, share ideas, or explore potential collaborations.
# 
# 
# #### **<h1 align="center"><span style="color:#690e11;">Contact Info</span>**
# ### ***Click on link below to contact/follow/correct me:***
# 
# - ***Email:*** mehakkhan301007@gmail.com
# - [LinkedIn](https://www.linkedin.com/in/mehak-iftikhar/)
# - [Facebook](https://www.facebook.com/profile.php?id=61552023122774)
# - [Twitter](https://twitter.com/mehakkhan874)
# - [Github](https://github.com/mehakiftikhar)

# # <p style="font-family:newtimeroman;font-size:150%;text-align:center;color:#690e11;">About Dataset</p>
# 
# ### ***`Dataset link:`*** [Fruit classification(10 Class)](https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class/data)
# 
# #### **<h1 align="left"><span style="color:#690e11;">Classes Of Dataset</span>**
# ### ***`Apple`***
# ### ***`Orange`***
# ### ***`Avocado`***
# ### ***`Kiwi`***
# ### ***`Mango`***
# ### ***`Pinenapple`***
# ### ***`Strawberries`***
# ### ***`Banana`***
# ### ***`Cherry`***
# ### ***`Watermelon`***
# 
# #### **<h1 align="center"><span style="color:#690e11;">SOURCES</span>**
# ### ***This data contains a set of images of 10 kind of fruits , that can be used to make classification using deep learning , i scraped this data from Instagram and google.***
# 
# #### **<h1 align="center"><span style="color:#690e11;">COLLECTION METHODOLOGY</span>**
# ### ***This data was collected by scraping my code in this rebo : https://github.com/karim-abdulnabi/CNN_model/tree/main/scraping_project class of data set : Apple Orange Avocado Kiwi Mango Pinenapple Strawberries Banana Cherry Watermelon i had collected 230 image for all kind from fruit for training and 110 for test***

# #### **<h1 align="center"><span style="color:#690e11;">Specifics</span>**
# ### ***Model: MobileNetV2***

# # **<p style="font-family:newtimeroman;font-size:150%;text-align:center;color:#690e11;">Import Libraries</p>**

# Utilisation de la librarie TensorFlow et Keras pour la création de modèles de visualisation et de classification d'images.
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import os
import random
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import warnings
import mlflow
import mlflow.keras
warnings.filterwarnings('ignore')


# ---

# ## ***train_datagen:***
# 
# ### ***`rescale=1./255:` Normalizes pixel values in the training images between 0 and 1. This is a common preprocessing step for image data, as neural networks typically work better with normalized values.***
# ### ***`rotation_range=40:` Applies random rotations to the training images within a range of -40° to 40° degrees. This helps the model become more robust to variations in object orientation.***
# ### ***`width_shift_range=0.1, height_shift_range=0.1:` Randomly shifts the training images horizontally and vertically by up to 10% of their width and height, respectively. This simulates small changes in camera position or object placement.***
# ### ***`horizontal_flip=True:` Randomly flips the training images horizontally. This helps the model learn to recognize objects regardless of their orientation.***
# ### ***`validation_split=0.2:` Splits the training data into training (80%) and validation (20%) sets. The validation set is used to monitor model performance during training and avoid overfitting.***
# 
# ## ***test_datagen:***
# 
# ### ***`rescale=1./255:` Applies the same normalization to the test images as the training images. This ensures consistency in the data processing pipeline.***

#Utilisation de MLflow pour le suivi des expériences de machine learning.
# Créer/utiliser une expérience (= projet de recherche)
mlflow.set_experiment("Fruit_Classification")


with mlflow.start_run():
    
    rescale = 1./255
    rotation_range=40
    width_shift_range=0.1
    height_shift_range=0.1
    horizontal_flip=True
    validation_split=0.2

    # Tracking des paramètres avec MLflow
    mlflow.log_param("rescale", rescale)
    mlflow.log_param("rotation_range", rotation_range)
    mlflow.log_param("width_shift_range", width_shift_range) 
    mlflow.log_param("height_shift_range", height_shift_range)
    mlflow.log_param("model_type", "MobileNetV2")
    mlflow.log_param("validation_split", validation_split)

    train_datagen = ImageDataGenerator(rescale = rescale, 
                                rotation_range=rotation_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                horizontal_flip=horizontal_flip,
                                validation_split=validation_split)

    val_datagen = ImageDataGenerator(rescale = rescale,
                                    validation_split=validation_split)
    test_datagen = ImageDataGenerator(rescale = rescale)


    # Chargement des images à partir de répertoires
    train_ds = train_datagen.flow_from_directory(
        directory = 'data/train',
        batch_size = 32,
        target_size = (224, 224),
        class_mode='categorical',
        subset="training",
        seed=123  
    )
    validation_ds = val_datagen.flow_from_directory(
        directory='data/train',
        batch_size=32,
        target_size=(224, 224),
        class_mode='categorical',
        subset="validation",
        seed=123 
    )
    test_ds = train_datagen.flow_from_directory(
        directory = 'data/test',
        batch_size = 32,
        target_size = (224, 224),
        class_mode='categorical'
    )

    # Model Building
    #Création d'un modèle de classification d'images en utilisant MobileNetV2 comme base pour le transfert learning.
    MobileNetV2_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3),pooling='avg')
    
    # Geler les couches de la base pré-entraînée pour éviter qu'elles ne soient mises à jour pendant l'entraînement
    MobileNetV2_base.trainable = False

    # Création du modèle séquentiel
    model = Sequential()

    # Ajout de la base pré-entraînée MobileNetV2 au modèle
    model.add(MobileNetV2_base)

    # Normalisation des activations pour stabiliser l'apprentissage et accélérer la convergence.
    model.add(BatchNormalization())

    # Abandon de 35% des neurones pour éviter le sur-apprentissage.
    model.add(Dropout(0.35))

    # Ajout d'une couche dense avec 220 unités et une fonction d'activation ReLU
    model.add(Dense(220, activation='relu'))

    # Ajout d'une couche dense avec 120 unités et une fonction d'activation ReLU
    model.add(Dense(60, activation='relu'))

    # Ajout de la couche de sortie avec 10 unités et une fonction d'activation softmax pour la classification multi-classes
    model.add(Dense(10, activation='softmax'))

    # Vérification du résumé du modèle
    model.summary()

    # Compilation du modèle
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss='categorical_crossentropy', metrics=['accuracy'])


    # Entraînement du modèle
    # Définition de la fonction de rappel
    early_stopping = EarlyStopping(patience=10)
    
    history= model.fit(train_ds,
        validation_data=validation_ds,
        steps_per_epoch=len(train_ds),
        epochs=2, 
        callbacks=[early_stopping]
)


    # Récupération des métriques d'entraînement et de validation
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    mlflow.log_metric("train_accuracy", train_accuracy[-1])
    mlflow.log_metric("val_accuracy", val_accuracy[-1])
    mlflow.log_metric("train_loss", train_loss[-1])
    mlflow.log_metric("val_loss", val_loss[-1])

    
    # Sauvegarder avec MLflow
    model_path = "mlflow/Fruit_Classification_model"
    mlflow.keras.log_model(model, "Fruit_Classification_model", registered_model_name="Fruit_Classification_model")
    mlflow.keras.save_model(model, model_path)
    print(f"✅ Modèle sauvegardé dans: {model_path}")