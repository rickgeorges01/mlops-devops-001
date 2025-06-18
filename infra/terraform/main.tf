# Acces vers les credentials aws
provider "aws" {
  region = "us-east-1"
  shared_credentials_files = ["../credit_aws/aws_learner_lab_credentials"]
  profile = "awslearnerlab"
}

# Recherche dynamique d'AMI pour Ubuntu 24.04 LTS
# Évite les identifiants AMI codés en dur
data "aws_ami" "ubuntu" {
  most_recent = true

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  owners = ["099720109477"] # Canonical
}

# Create an EC2 instance
resource "aws_instance" "example" {
  ami                    = data.aws_ami.ubuntu.id
  key_name               = "vockey"
  instance_type          = "t2.micro"
  vpc_security_group_ids = [aws_security_group.allow_ssh.id, aws_security_group.allow_http_s.id]

  tags = {
    Name = "predi"
  }
}