# Terraform stub

Minimal, intentionally-not-deployed IaC proving out the cloud/containerized
deployment shape for the serving API — see `project-brief.md`'s IaC item and
explicit non-goal ("no real cloud deployment required").

**Scope:** an ECR repository plus a single-task Fargate service running the
`api` image, using the target account's default VPC (no networking of its
own — this is a stub, not a second infrastructure project). The
Airflow/MLflow/Postgres/dashboard side of the stack is deliberately out of
scope here; those run via `infra/docker-compose.yml` locally, and a real
production version would move them to managed equivalents (RDS for Postgres,
a managed orchestrator, a managed MLflow tracking server, etc.).

**Not applied as part of this project.** To actually plan/apply against a
real AWS account:

```
cd infra/terraform
terraform init
terraform plan -var="aws_region=us-east-1"
```
