# Minimal, coherent IaC stub proving out the cloud/containerized deployment
# shape for the serving API - see project-brief.md's IaC item ("doesn't need
# to be deployed live, needs to exist and be coherent"). Scoped to the api
# service only, the component the brief calls out for "cloud/containerized
# deployment" - not the full Airflow/MLflow/Postgres/dashboard stack, which
# infra/docker-compose.yml runs locally and a real production version would
# put on managed equivalents (see terraform/README.md).
#
# Uses the account's default VPC/subnets rather than provisioning networking
# of its own, to keep this a stub instead of a second infrastructure project.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-cluster"
}

resource "aws_iam_role" "execution" {
  name = "${var.project_name}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.execution.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.container_image_tag}"
      essential = true
      portMappings = [
        { containerPort = var.container_port, protocol = "tcp" }
      ]
      # A real deployment would inject DATABASE_URL / MLFLOW_TRACKING_URI here
      # from Secrets Manager or SSM Parameter Store, pointing at RDS and a
      # managed MLflow endpoint - wiring those is out of scope for this stub.
    }
  ])
}

resource "aws_security_group" "api" {
  name        = "${var.project_name}-api"
  description = "Placeholder - allows inbound API traffic directly; a real deployment would put a load balancer in front and scope this to it, not 0.0.0.0/0"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }
}
