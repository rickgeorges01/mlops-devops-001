#!/bin/bash

echo " [MLPROJECT] ---> Démarrage du projet MLOps Fruit Classifier"
echo "============================================="

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo " [ERROR] ---> Docker n'est pas installé"
    echo " [INFO] ---> Installez Docker Desktop et redémarrez"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo " [ERROR] ---> Docker-compose n'est pas installé"
    echo " [INFO] ---> Installez docker compose et redémarrez"
    exit 1
fi

# Vérifier que les fichiers nécessaires existent
if [ ! -f "docker-compose-api.yml" ]; then
    echo " [ERROR] ---> Fichier docker-compose-api.yml non trouvé"
    echo " [INFO] ---> Assurez-vous d'être dans le bon répertoire"
    exit 1
fi

if [ ! -f "docker-compose-training.yml" ]; then
    echo " [ERROR] ---> Fichier docker-compose-training.yml non trouvé"
    exit 1
fi

if [ ! -d "data/test" ]; then
    echo " [ERROR] ---> Dossier data/test non trouvé"
    echo " [INFO] ---> Le dataset est nécessaire pour les prédictions"
    exit 1
fi

# Arrêter les anciens containers
echo " [INFO] ---> Nettoyage des anciens containers..."
docker compose -f docker-compose-api.yml down 2>/dev/null
docker compose -f docker-compose-training.yml down 2>/dev/null

# Nettoyer les images pour éviter les conflits
echo " [INFO] ---> Nettoyage des images Docker..."
docker system prune -f

# Étape 1 : Lancer API + MLflow
echo " [INFO] ---> Lancement des services API + MLflow..."
docker compose -f docker-compose-api.yml up -d --build

# Attendre que MLflow soit prêt
echo " [INFO] ---> Attente du démarrage de MLflow..."
for i in {1..30}; do
    if curl -s http://localhost:5001 > /dev/null 2>&1; then
        echo " [CORRECT] ---> MLflow est prêt !"
        break
    fi
    echo "   Tentative $i/30..."
    sleep 2
done

# Vérifier que MLflow est accessible
if ! curl -s http://localhost:5001 > /dev/null 2>&1; then
    echo " [ERREUR] ---> MLflow n'est pas accessible"
    echo " [INFO] ---> État des containers :"
    docker compose -f docker-compose-api.yml ps
    exit 1
fi

# Étape 2 : Entraînement du modèle
echo " [INFO] ---> Lancement de l'entraînement du modèle..."
echo " [INFO] ---> Cela peut prendre 8-10 minutes..."

# Configurer la variable pour pointer vers l'API locale
export API_INSTANCE_IP=host.docker.internal

docker compose -f docker-compose-training.yml up --build

# Attendre que l'entraînement se termine
echo " [INFO] --->  Attente de la fin de l'entraînement..."
sleep 10

# Arrêter le container training
echo " [INFO] ---> Arrêt du container d'entraînement..."
docker compose -f docker-compose-training.yml down

# Vérifier l'état final des services
echo " [INFO] ---> État final des services :"
docker compose -f docker-compose-api.yml ps

# Tests de connectivité
echo ""
echo " [INFO] ---> Tests de connectivité :"

if curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo " [CORRECT] ---> API accessible"
else
    echo " [ERROR] ---> API non accessible"
fi

if curl -s http://localhost:5001 > /dev/null 2>&1; then
    echo " [CORRECT] ---> MLflow accessible"
else
    echo " [ERROR] ---> MLflow non accessible"
fi

echo ""
echo "Démarrage terminé !"
echo "======================================"
echo "Interface API : http://localhost:8000"
echo "MLflow UI : http://localhost:5001"
echo ""
echo "Instructions :"
echo "1. Ouvrez http://localhost:8000 dans votre navigateur"
echo "2. Testez les prédictions avec :"
echo "   - Option 1 : Images du dataset (dropdown)"
echo "   - Option 2 : Upload de vos propres images"
echo ""
echo "Pour arrêter les services :"
echo "  docker-compose -f docker-compose-api.yml down"
echo ""
echo "Pour voir les logs :"
echo "  docker-compose -f docker-compose-api.yml logs -f"