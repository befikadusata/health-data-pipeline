variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix used to name every resource in this stub"
  type        = string
  default     = "health-pipeline"
}

variable "container_port" {
  description = "Port the api container listens on (see infra/Dockerfile.app)"
  type        = number
  default     = 8000
}

variable "container_image_tag" {
  description = "Image tag to deploy - a real pipeline would set this from CI on each build, not a static default"
  type        = string
  default     = "latest"
}
