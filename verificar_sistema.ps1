# Script para verificar la configuración y estado del sistema
# filepath: d:\Escuela\Redes\Proyecto Distribuido\scr_pasd_2025\verificar_sistema.ps1

param(
    [Parameter()]
    [ValidateSet('local', 'docker')]
    [string]$entorno = "local"
)

# Navegar al directorio del proyecto
$projectDir = $PSScriptRoot
Set-Location $projectDir

function Test-Dependencies {
    Write-Host "Verificando dependencias instaladas..." -ForegroundColor Cyan
    
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
    } 
    catch {
        Write-Host "✗ Python no encontrado. Por favor instala Python 3.9 o superior." -ForegroundColor Red
        return $false
    }
    
    try {
        $packages = pip list
        $rayInstalled = $packages | Select-String -Pattern "ray" -Quiet
        
        if ($rayInstalled) {
            Write-Host "✓ Ray instalado" -ForegroundColor Green
        } 
        else {
            Write-Host "✗ Ray no instalado. Ejecuta 'pip install -r requirements.txt'" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ Error al verificar paquetes pip" -ForegroundColor Red
        return $false
    }
    
    if ($entorno -eq "docker") {
        try {
            $dockerVersion = docker --version 2>&1
            Write-Host "✓ Docker: $dockerVersion" -ForegroundColor Green
            
            $dockerComposeVersion = docker compose version 2>&1
            Write-Host "✓ Docker Compose: $dockerComposeVersion" -ForegroundColor Green
        } 
        catch {
            Write-Host "✗ Docker o Docker Compose no encontrado. Instala Docker Desktop." -ForegroundColor Red
            return $false
        }
    }
    
    return $true
}

function Test-Configuration {
    Write-Host "Verificando archivos de configuración..." -ForegroundColor Cyan
      if (Test-Path "$projectDir\src\config\config.yaml") {
        Write-Host "✓ Archivo de configuración encontrado" -ForegroundColor Green
    } 
    else {
        Write-Host "✗ Archivo de configuración no encontrado. Creando archivo de configuración..." -ForegroundColor Yellow
        
        # Crear un archivo de configuración básico
        @"
###### Archivo de configuración para la Plataforma de Aprendizaje Supervisado Distribuido

# Configuración de Entrenamiento
datasets:
  - name: iris
    target_column: species
    test_size: 0.2
    random_state: 42
  
  - name: diabetes
    target_column: target
    test_size: 0.25
    random_state: 42

# Configuración de Modelos
models:
  - type: random_forest
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
      task: classification
  
  - type: gradient_boosting
    params:
      n_estimators: 100
      learning_rate: 0.1
      random_state: 42
      task: classification
  
  - type: logistic_regression
    params:
      max_iter: 1000
      random_state: 42
  
  - type: svm
    params:
      kernel: rbf
      C: 1.0
      random_state: 42
      task: classification

# Configuración de Ray
ray:
  head_address: localhost:6379
  redis_password: null
  num_cpus: null  # Usar todas las CPUs disponibles
  num_gpus: 0

# Configuración de Servicio
serving:
  port: 8000
  host: 0.0.0.0

# Configuración de Monitoreo
monitoring:
  interval: 5  # en segundos
  save_plots: true
"@ | Out-File -FilePath "$projectDir\src\config\config.yaml" -Encoding utf8
        
        Write-Host "✓ Archivo de configuración creado" -ForegroundColor Green
    }
    
    # Verificar directorios de datos
    if (-not (Test-Path "$projectDir\data\raw")) {
        New-Item -ItemType Directory -Path "$projectDir\data\raw" -Force | Out-Null
        Write-Host "✓ Directorio de datos creado" -ForegroundColor Green
    }
    
    # Verificar directorios de modelos
    if (-not (Test-Path "$projectDir\models")) {
        New-Item -ItemType Directory -Path "$projectDir\models" -Force | Out-Null
        Write-Host "✓ Directorio de modelos creado" -ForegroundColor Green
    }
    
    # Verificar directorios de gráficos
    if (-not (Test-Path "$projectDir\plots")) {
        New-Item -ItemType Directory -Path "$projectDir\plots" -Force | Out-Null
        Write-Host "✓ Directorio de gráficos creado" -ForegroundColor Green
    }
}

function Test-DataGeneration {
    Write-Host "Verificando datos de ejemplo..." -ForegroundColor Cyan
    
    if (-not (Test-Path "$projectDir\data\raw\iris.csv")) {
        Write-Host "Generando datos de ejemplo..." -ForegroundColor Yellow
        try {
            python "$projectDir\generate_data.py"
            if (Test-Path "$projectDir\data\raw\iris.csv") {
                Write-Host "✓ Datos generados correctamente" -ForegroundColor Green
            }
            else {
                Write-Host "✗ Error al generar datos" -ForegroundColor Red
                return $false
            }
        }
        catch {
            Write-Host "✗ Error al ejecutar generate_data.py: $_" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "✓ Los datos de ejemplo ya existen" -ForegroundColor Green
    }
    
    return $true
}

function Test-DockerImages {
    if ($entorno -ne "docker") {
        return $true
    }
    
    Write-Host "Verificando imágenes Docker..." -ForegroundColor Cyan
    
    $imageExists = docker images -q scr-pasd-system 2>$null
    
    if (-not $imageExists) {
        Write-Host "Construyendo imagen Docker..." -ForegroundColor Yellow
        try {
            docker-compose -f docker/docker-compose.yml build
            $exitCode = $LASTEXITCODE
            
            if ($exitCode -eq 0) {
                Write-Host "✓ Imágenes Docker construidas correctamente" -ForegroundColor Green
            }
            else {
                Write-Host "✗ Error al construir imágenes Docker (código $exitCode)" -ForegroundColor Red
                return $false
            }
        }
        catch {
            Write-Host "✗ Error al construir imágenes Docker: $_" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "✓ Imágenes Docker ya existen" -ForegroundColor Green
    }
    
    return $true
}

Write-Host "=== Verificación del Sistema ===" -ForegroundColor Magenta
Write-Host "Entorno: $entorno" -ForegroundColor Magenta

$dependenciesOk = Test-Dependencies
if (-not $dependenciesOk) {
    Write-Host "Hay problemas con las dependencias. Por favor, resuélvalos antes de continuar." -ForegroundColor Red
    exit 1
}

Test-Configuration
$dataOk = Test-DataGeneration
if (-not $dataOk) {
    Write-Host "Hay problemas con la generación de datos. Por favor, resuélvalos antes de continuar." -ForegroundColor Red
    exit 1
}

if ($entorno -eq "docker") {
    $dockerOk = Test-DockerImages
    if (-not $dockerOk) {
        Write-Host "Hay problemas con las imágenes Docker. Por favor, resuélvalos antes de continuar." -ForegroundColor Red
        exit 1
    }
}

Write-Host "=== Verificación Completada ===" -ForegroundColor Magenta
Write-Host "El sistema está correctamente configurado para el entorno '$entorno'" -ForegroundColor Green
Write-Host ""
Write-Host "Para ejecutar el sistema en este entorno:" -ForegroundColor Cyan
if ($entorno -eq "docker") {
    Write-Host "   .\ejecutar_docker.ps1 -componente todo" -ForegroundColor Yellow
} else {
    Write-Host "   .\ejecutar_local.ps1 -modo todo" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Para más información, consulta QUICK_START.md" -ForegroundColor Cyan
