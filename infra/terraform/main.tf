# Access vers les credentials aws
provider "aws" {
  region = var.region
  shared_credentials_files = ["../credit_aws/aws_learner_lab_credentials"]
  profile = "awslearnerlab"
}

#  Data source pour récupérer l'AMI Ubuntu la plus récente
data "aws_ami" "ubuntu" {
    most_recent = true
    owners = ["099720109477"] # Canonical

    filter {
        name = "name"
        values = ["ubuntu/images/hvm-ssd/ubuntu-24.04-amd64-server-*"]
    }

    filter {
        name = "virtualization-type"
        values = ["hvm"]
    }

    filter {
        name = "state"
        values = ["available"]
    }
}


# Security Group pour l'instance API
resource "aws_security_group" "api_security_group" {
       name = "${var.project_name}-api-sg"
       description = "Security group pour l'instance API (FastApi +MLflow)"

       # SSH depuis internet
       ingress {
        description = "SSH"
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
       }

       # Fast API depuis internet
       ingress {
        description = "Fast API"
        from_port = 8000
        to_port = 8000
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
       }

       # Communication MLflow depuis instance Training
       ingress {
        description = "MLflow depuis Training"
        from_port = "5001"
        to_port = "5001"
        protocol = "tcp"
        security_groups = [aws_security_group.training_security_group.id]
       }

       # Autorisation de tout les trafics sortants
       egress {
         from_port = 0
         to_port = 0
         protocol = "-1"
         cidr_blocks = ["0.0.0.0/0"]
       }

       tags = {
         Name = "${var.project_name}-api-sg"
       }
}
resource "aws_security_group" "training_security_group"{
    name = "${var.project_name}-training-sg"
    description = "Security group pour l'instance training"
    # SSH depuis internet
    ingress{
        description = "SSH"
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["0.0.0.0/0"]
    }

    # Autorisation tout les trafic sortants
    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags = {
         Name = "${var.project_name}-training-sg"
    }
}

resource "aws_ebs_volume" "mlflow-data" {
  availability_zone = data.aws_availability_zones.available.names[0]
  size = var.ebs_volume_size
  type  = "gp3"
  encrypted = true

  tags = {
    Name = "${var.project_name}-mlflow-data"
  }
}


# Data source pour les zones de disponibilité
data "aws_availability_zones" "available" {
    state = "available"
}


# Instance EC2 pour l'API
resource "aws_instance" "api_instance" {
  ami                     = data.aws_ami.ubuntu.id
  instance_type           = var.api_instance_type
  key_name                = var.key_name
  availability_zone       = data.aws_availability_zones.available.names[0]
  vpc_security_group_ids  = [aws_security_group.api_security_group.id]

  # Script d'initialisation
  user_data = base64encode(templatefile("${path.module}/user-data-api.sh", {
    git_repo_url     = var.git_repo_url
    project_name     = var.project_name
    ebs_device       = "/dev/xvdf"
    mount_point      = "/opt/mlflow-data"
  }))

  # Configuration du stockage root
  root_block_device {
    volume_size = 20  # 20GB pour le système + Docker
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "${var.project_name}-api-instance"
    Type = "API-Server"
  }

  # Instance est crée avant d'attacher le volume
  depends_on = [aws_ebs_volume.mlflow-data]
}


# Attachement du volume EBS à l'instance API
resource "aws_volume_attachment" "mlflow_data_attachment" {
    device_name = "/dev/xvdf"
    volume_id = aws_ebs_volume.mlflow-data.id,
    instance_id = aws_instance.api_instance.id
}


# Instance EC2 pour le Training
resource "aws_instance" "training_instance" {
  ami                     = data.aws_ami.ubuntu.id
  instance_type           = var.training_instance_type
  key_name                = var.key_name
  vpc_security_group_ids  = [aws_security_group.training_security_group.id]

  # Script d'initialisation
  user_data = base64encode(templatefile("${path.module}/user-data-training.sh", {
    git_repo_url     = var.git_repo_url
    project_name     = var.project_name
    api_instance_ip  = aws_instance.api_instance.private_ip
  }))

  # Configuration du stockage root
  root_block_device {
    volume_size = 25  # Plus d'espace pour les datasets
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "${var.project_name}-training-instance"
    Type = "Training-Server"
  }

  # S'assurer que l'instance API est créée en premier
  depends_on = [aws_instance.api_instance]
}