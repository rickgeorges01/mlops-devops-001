#!/bin/bash

# Script d'initialisation pour l'instance API
# Installe Docker, clone le repo, configure et lance les services

set -e
# Redirection des logs vers un dossier afin de faciliter son accessibilité
# - via log complets (user-data.log)
# - via log système (logger)
# - via console aws EC2 (console)
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Début de l'initialisation de l'instance API ==="
echo "Timestamp: $(date)"

# Variables passées par Terraform
GIT_REPO_URL="${git_repo_url}"
PROJECT_NAME="${project_name}"
EBS_DEVICE="${ebs_device}"
MOUNT_POINT="${mount_point}"

echo "Configuration:"
echo " [INFO] ---> Git Repo: $GIT_REPO_URL"
echo " [INFO] ---> Project: $PROJECT_NAME"
echo " [INFO] ---> EBS Device: $EBS_DEVICE"
echo " [INFO] ---> Mount Point: $MOUNT_POINT"

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

# Attendre que le volume EBS soit attaché
echo "=== Configuration du volume EBS ==="
sleep 30

# Vérifier si le volume est attaché
while [ ! -e $EBS_DEVICE ]; do
    echo "Attente du volume EBS $EBS_DEVICE..."
    sleep 10
done

# Créer le système de fichiers s'il n'existe pas
if ! blkid $EBS_DEVICE; then
    echo " [INFO] ---> Création du système de fichiers sur $EBS_DEVICE"
    mkfs.ext4 $EBS_DEVICE
fi

# Créer le point de montage et monter le volume
mkdir -p $MOUNT_POINT
mount $EBS_DEVICE $MOUNT_POINT

# Configuration du montage automatique
echo "$EBS_DEVICE $MOUNT_POINT ext4 defaults,nofail 0 2" >> /etc/fstab

# Permissions pour l'utilisateur ubuntu
chown ubuntu:ubuntu $MOUNT_POINT

# Clonage du repository
echo "=== Clonage du repository ==="
cd /home/ubuntu
sudo -u ubuntu git clone $GIT_REPO_URL $PROJECT_NAME
chown -R ubuntu:ubuntu /home/ubuntu/$PROJECT_NAME

# Configuration des variables d'environnement
echo "=== Configuration des variables d'environnement ==="
cat >> /home/ubuntu/.bashrc << EOF

# Variables pour le projet MLOps
export MLFLOW_DATA_PATH=$MOUNT_POINT
export PROJECT_PATH=/home/ubuntu/$PROJECT_NAME
export MLFLOW_TRACKING_URI=http://localhost:5001

EOF

# Lancement des services
echo "=== Préparation du lancement des services ==="
cd /home/ubuntu/$PROJECT_NAME

# Créer le script de démarrage
cat > /home/ubuntu/start-services.sh << 'SCRIPT'
#!/bin/bash
cd /home/ubuntu/mlops-fruit-classifier
echo "Démarrage des services API..."
docker-compose -f docker-compose-api.yml up -d --build
echo "Services démarrés !"
docker compose -f docker-compose-api.yml ps
SCRIPT

# Rend le fichier exécutable
chmod +x /home/ubuntu/start-services.sh
chown ubuntu:ubuntu /home/ubuntu/start-services.sh

# Lancer les services en tant qu'ubuntu
echo "=== Lancement des services ==="
sudo -u ubuntu /home/ubuntu/start-services.sh

# Configuration des logs
echo "=== Configuration des logs ==="
mkdir -p /var/log/$PROJECT_NAME
chown ubuntu:ubuntu /var/log/$PROJECT_NAME

echo "=== Installation terminée ! ==="
echo " [STATUS] ---> Services démarrés :"
echo " [INFO] ---> API sur le port 8000"
echo " [INFO] ---> MLflow sur le port 5001"
echo ""
echo " [INFO] ---> Pour les URLs complètes, lancez :"
echo " [ACTION] ---> terraform output service_urls"