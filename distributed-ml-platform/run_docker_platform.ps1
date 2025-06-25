# Run the distributed ML platform in Docker
Write-Host "Starting the Distributed ML Platform using Docker..." -ForegroundColor Green

# Pull necessary images
Write-Host "Pulling latest Python image..." -ForegroundColor Cyan
docker pull python:3.9-slim

# Build and start the containers
Write-Host "Building and starting the containers..." -ForegroundColor Cyan
docker-compose up --build -d

# Function to check container logs
function Get-ContainerLogs {
    param (
        [string]$containerName,
        [int]$lines = 20
    )
    
    Write-Host "`n======== $containerName Logs ========" -ForegroundColor Yellow
    docker logs --tail $lines $containerName
}

# Wait a moment for containers to initialize
Write-Host "Waiting for services to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Check if containers are running
$containers = @("distributed-ml-platform_ray-head_1", "distributed-ml-platform_ray-worker-1_1", "distributed-ml-platform_ray-worker-2_1")
foreach ($container in $containers) {
    $status = docker ps --filter "name=$container" --format "{{.Status}}"
    
    if ($status) {
        Write-Host "Container $container is running: $status" -ForegroundColor Green
    } else {
        Write-Host "Container $container is NOT running!" -ForegroundColor Red
    }
}

# Show logs for each container
foreach ($container in $containers) {
    Get-ContainerLogs -containerName $container -lines 30
}

Write-Host "`nDistributed ML Platform is running!" -ForegroundColor Green
Write-Host "- API is available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "- Ray Dashboard is available at: http://localhost:8265" -ForegroundColor Cyan
Write-Host "`nTo see more logs:" -ForegroundColor Cyan
Write-Host "docker logs -f distributed-ml-platform_ray-head_1" -ForegroundColor Gray
Write-Host "`nTo stop the platform:" -ForegroundColor Cyan
Write-Host "docker-compose down" -ForegroundColor Gray
