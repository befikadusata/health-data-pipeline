output "ecr_repository_url" {
  description = "Push target for the api image, e.g. from CI: docker push <this>:<tag>"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}
