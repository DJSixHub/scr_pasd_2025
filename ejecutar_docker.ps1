# Script para ejecutar el proyecto con Docker

param(
    [Parameter()]
    [ValidateSet('todo', 'head', 'worker', 'serving', 'monitor')]
    [string]$componente = "todo",
    
    [Parameter()]
    [string]$configPath = "src/config/config.yaml"
)

# Navegar al directorio del proyecto
$projectDir = $PSScriptRoot
Set-Location $projectDir

# Construir las imágenes Docker
Write-Host "Construyendo imágenes Docker..." -ForegroundColor Cyan
docker-compose -f docker/docker-compose.yml build --no-cache

# Verificar si la construcción fue exitosa
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error al construir las imágenes Docker. Saliendo..." -ForegroundColor Red
    exit 1
}

# Crear un archivo de entorno para pasar la configuración a Docker
$envFilePath = "$projectDir\docker\.env"
"CONFIG_FILE=$configPath" | Out-File -FilePath $envFilePath -Encoding utf8 -Force

# Ejecutar los servicios según el componente especificado
Write-Host "Iniciando servicios: $componente (usando configuración: $configPath)..." -ForegroundColor Green

switch ($componente) {
    "todo" {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up
    }
    "head" {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up head
    }
    "worker" {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up worker
    }
    "serving" {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up serving
    }
    "monitor" {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up monitor
    }
    default {
        docker-compose -f docker/docker-compose.yml --env-file docker/.env up
    }
}