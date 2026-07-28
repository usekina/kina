# KinaBot Simple AWS Deployment

This first deployment is intentionally small and independent from Term1:

```text
Docker image
  → ECR
  → one ECS Fargate task
  → Application Load Balancer URL
  → encrypted EFS for scores and the local Whisper model
```

CloudWatch receives application logs and EFS backup is enabled. All named
resources use the `kinabot-` prefix.

## First Private Test

The initial stack runs in staging mode:

- a user enters an email and six-digit code on the same page;
- no password is created or remembered;
- OpenAI is not configured;
- no domain or certificate is required; and
- the ALB supplies a temporary stable HTTP URL.

During private infrastructure testing, the code can appear on screen. Before
sharing the URL, connect Amazon SES so the code is delivered to the entered
email and each verified email receives up to two analyses per day.

## Deploy

Prerequisites:

```powershell
aws login
```

Start Docker Desktop and wait until its engine is running. Then:

```powershell
cd "C:\Users\648493\OneDrive - TMNA\Desktop\Term1\kina_worktree\aoi_kinabot_app"
.\aws\deploy.ps1
```

The script:

1. confirms AWS and Docker;
2. creates the separate `kinabot` ECR repository if needed;
3. builds and pushes the container;
4. finds two public subnets in the default VPC;
5. deploys the independent `kinabot-production` stack; and
6. prints the URL.

## Before Inviting Users

After the app works from the AWS URL:

1. connect Amazon SES email-code delivery;
2. disable on-screen staging codes and add HTTPS;
3. generate a QR code for the AWS ALB URL;
4. test data deletion and backup recovery; and
5. invite 5–10 users before expanding.

The first pilot uses one ECS task and EFS-backed SQLite. Keep the desired task
count at one. Migrate to RDS PostgreSQL before horizontal scaling.
