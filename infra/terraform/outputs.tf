# Outputs - Informations importantes après déploiement

output "api_instance_info" {
  description = "Informations de l'instance API"
  value = {
    public_ip  = aws_instance.api_instance.public_ip
    private_ip = aws_instance.api_instance.private_ip
    instance_id = aws_instance.api_instance.id
  }
}

output "training_instance_info" {
  description = "Informations de l'instance Training"
  value = {
    public_ip  = aws_instance.training_instance.public_ip
    private_ip = aws_instance.training_instance.private_ip
    instance_id = aws_instance.training_instance.id
  }
}

output "service_urls" {
  description = "URLs des services déployés"
  value = {
    api_url     = "http://${aws_instance.api_instance.public_ip}:8000"
    mlflow_ui   = "http://${aws_instance.api_instance.public_ip}:5001"
  }
}

output "ssh_commands" {
  description = "Commandes SSH pour se connecter aux instances"
  value = {
    api_instance = "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.api_instance.public_ip}"
    training_instance = "ssh -i ${var.key_name}.pem ubuntu@${aws_instance.training_instance.public_ip}"
  }
}

output "ebs_volume_info" {
  description = "Informations du volume EBS MLflow"
  value = {
    volume_id = aws_ebs_volume.mlflow-data.id
    size_gb   = aws_ebs_volume.mlflow-data.size
    attached_to = aws_instance.api_instance.id
  }
}