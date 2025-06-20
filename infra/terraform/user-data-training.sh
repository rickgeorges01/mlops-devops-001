#!/bin/bash

# Script d'initialisation pour l'instance Training
# Installe Docker, clone le repo, configure et lance le service training

set -e
# Redirection des logs vers un dossier afin de faciliter son accessibilité
# - via log complets (user-data.log)
# - via log système (logger)
# - via console aws EC2 (console)
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Début de l'initialisation de l'instance Training ==="
echo "Timestamp: $(date)"

# Variables passées par Terraform
GIT_REPO_URL="${git_repo_url}"
PROJECT_NAME="${project_name}"
API_INSTANCE_IP="${api_instance_ip}"

echo "Configuration:"
echo " [INFO] ---> Git Repo: $GIT_REPO_URL"
echo " [INFO] ---> Project: $PROJECT_NAME"
echo " [INFO] ---> API Instance IP: $API_INSTANCE_IP"

# Mise à jour du système
echo "=== Mise à jour du système ==="
apt-get update
apt-get upgrade -y

# Installation des dépendances de base
echo "=== Installation des dépendances ==="
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common \
    git \
    unzip

# Installation de Docker
echo "=== Installation de Docker ==="
# Téléchargement et sécurisation de la clé publique de Docker afin de permettre son téléchargement
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
#  Ajout de Docker à votre système Ubuntu.
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Configuration Docker
systemctl start docker
systemctl enable docker
usermod -aG docker ubuntu

# Clonage du repository
echo "=== Clonage du repository ==="
cd /home/ubuntu
sudo -u ubuntu git clone $GIT_REPO_URL $PROJECT_NAME
chown -R ubuntu:ubuntu /home/ubuntu/$PROJECT_NAME

# Configuration des variables d'environnement
echo "=== Configuration des variables d'environnement ==="
cat >> /home/ubuntu/.bashrc << EOF

# Variables pour le projet MLOps Training
export API_INSTANCE_IP=$API_INSTANCE_IP
export MLFLOW_TRACKING_URI=http://$API_INSTANCE_IP:5001
export PROJECT_PATH=/home/ubuntu/$PROJECT_NAME

EOF

# Création du fichier .env pour docker-compose
echo "=== Création du fichier .env ==="
cat > /home/ubuntu/$PROJECT_NAME/.env << EOF
# Configuration pour l'instance Training
MLFLOW_TRACKING_URI=http://$API_INSTANCE_IP:5001
PYTHONUNBUFFERED=1
API_INSTANCE_IP=$API_INSTANCE_IP
EOF

chown ubuntu:ubuntu /home/ubuntu/$PROJECT_NAME/.env

# Lancement du service training en mode attente
echo "=== Préparation du service training ==="
cd /home/ubuntu/$PROJECT_NAME

# Créer le script de démarrage du training
cat > /home/ubuntu/start-training-container.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/$PROJECT_NAME
echo "Démarrage du container training en mode attente..."
docker compose -f docker-compose-training.yml up -d --build
echo "Container training démarré en mode attente !"
docker compose -f docker-compose-training.yml ps
SCRIPT

chmod +x /home/ubuntu/start-training-container.sh
chown ubuntu:ubuntu /home/ubuntu/start-training-container.sh

# Créer le script pour lancer l'entraînement
cat > /home/ubuntu/run-training.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/$PROJECT_NAME
echo "Lancement de l'entraînement du modèle..."
docker compose -f docker-compose-training.yml exec -T training python train.py
echo "Entraînement terminé !"
SCRIPT

chmod +x /home/ubuntu/run-training.sh
chown ubuntu:ubuntu /home/ubuntu/run-training.sh

# Lancer le container training en mode attente
echo "=== Lancement du container training ==="
sudo -u ubuntu /home/ubuntu/start-training-container.sh

# Attendre que l'API soit accessible avant de tester la connexion
echo "=== Test de connectivité vers l'instance API ==="
sleep 60

# Test de connexion vers l'API MLflow
if curl -f http://$API_INSTANCE_IP:5001 > /dev/null 2>&1; then
    echo " [INFO] ---> Connexion vers MLflow API réussie"
else
    echo " [ERROR] ---> Connexion vers MLflow API échouée - vérifiez que l'instance API est démarrée"
fi

# Configuration des logs
echo "=== Configuration des logs ==="
mkdir -p /var/log/$PROJECT_NAME
chown ubuntu:ubuntu /var/log/$PROJECT_NAME

echo "=== Fin de l'initialisation de l'instance Training ==="
echo "Timestamp: $(date)"
echo " [INFO] ---> Instance Training configurée et prête !"
echo ""
echo " [ACTION] ---> Commandes utiles après connexion SSH :"
echo "   cdp            --> Aller dans le projet"
echo "   run-training   --> Lancer un entraînement"
echo "   training-logs  --> Voir les logs"
echo ""
echo " [INFO] ---> Connexion MLflow : http://$API_INSTANCE_IP:5001"