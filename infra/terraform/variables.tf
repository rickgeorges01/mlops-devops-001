# Variables pour l'infrastructure MLOp
variable "region" {
    description = "Region AWS ou deployer notre infrastructure"
    type = string
    default = "us-east-1"
}

variable "api_instance_type" {
    description = "Type d'instance pour le serveur API (Fast +MLflow)"
    type = string
    default = "t3.micro"
}

variable "training_instance_type" {
    description = "Type d'instance pour le training"
    type = string
    default = "t3.large"
}

variable "git_repo_url" {
    description = "URL du repository Git contenant le code de l'application"
    type = string
    default = "https://github.com/rickgeorges01/mlops-devops-001.git"
}

variable "key_name" {
    description = "Nom de la clé SSH AWS pour accéder aux instances"
    type = string
    default = "welovemlops-devops"
}

variable "project_name" {
  description = "Nom du projet"
  type        = string
  default     = "mlops-fruit-classifier"
}

variable "ebs_volume_size" {
  description = "Taille du volume EBS pour le stockage MLflow (en GB)"
  type        = number
  default     = 10
}