param(
    [string]$Region = "us-west-2",
    [string]$StackName = "kinabot-production",
    [string]$RepositoryName = "kinabot",
    [string]$CertificateArn = ""
)

$ErrorActionPreference = "Stop"

$identity = aws sts get-caller-identity --region $Region | ConvertFrom-Json
if (-not $identity.Account) {
    throw "AWS authentication is not available. Run 'aws login' first."
}

docker version | Out-Null

$accountId = $identity.Account
$registry = "$accountId.dkr.ecr.$Region.amazonaws.com"
$imageUri = "$registry/$RepositoryName`:latest"

$previousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
aws ecr describe-repositories `
    --repository-names $RepositoryName `
    --region $Region 2>$null | Out-Null
$repositoryExists = $LASTEXITCODE -eq 0
$ErrorActionPreference = $previousErrorPreference
if (-not $repositoryExists) {
    aws ecr create-repository `
        --repository-name $RepositoryName `
        --image-scanning-configuration scanOnPush=true `
        --encryption-configuration encryptionType=AES256 `
        --region $Region | Out-Null
}

aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $registry | Out-Null

$appDirectory = Split-Path -Parent $PSScriptRoot
docker build --pull --tag $imageUri $appDirectory
docker push $imageUri

$vpcId = aws ec2 describe-vpcs `
    --filters Name=is-default,Values=true `
    --query "Vpcs[0].VpcId" `
    --output text `
    --region $Region
if (-not $vpcId -or $vpcId -eq "None") {
    throw "No default VPC found. Pass an approved VPC by adapting this deployment."
}

$subnets = @(
    aws ec2 describe-subnets `
        --filters "Name=vpc-id,Values=$vpcId" `
        --query "Subnets[?MapPublicIpOnLaunch==``true``].SubnetId" `
        --output text `
        --region $Region
) -split "\s+"
$subnets = @($subnets | Where-Object { $_ })
if ($subnets.Count -lt 2) {
    throw "At least two public subnets are required for the load balancer."
}

$parameterOverrides = @(
    "VpcId=$vpcId",
    "SubnetA=$($subnets[0])",
    "SubnetB=$($subnets[1])",
    "ContainerImage=$imageUri"
)
if ($CertificateArn) {
    $parameterOverrides += "CertificateArn=$CertificateArn"
}

aws cloudformation deploy `
    --template-file (Join-Path $PSScriptRoot "cloudformation.yml") `
    --stack-name $StackName `
    --capabilities CAPABILITY_NAMED_IAM `
    --parameter-overrides $parameterOverrides `
    --tags Application=KinaBot Environment=production `
    --region $Region

aws ecs update-service `
    --cluster kinabot-production `
    --service kinabot-production `
    --force-new-deployment `
    --region $Region | Out-Null

aws cloudformation describe-stacks `
    --stack-name $StackName `
    --query "Stacks[0].Outputs" `
    --output table `
    --region $Region
