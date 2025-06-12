# Script para ejecutar el proyecto sin Docker

param(
    [Parameter()]
    [ValidateSet('todo', 'entrenamiento', 'servicio', 'monitoreo')]
    [string]$modo = "todo",
    
    [Parameter()]
    [string]$configPath = "src/config/config.yaml"
)

# Navegar al directorio del proyecto
$projectDir = $PSScriptRoot
Set-Location $projectDir

# Primero, generar los datos de ejemplo si no existen
if (-not (Test-Path "$projectDir\data\raw\iris.csv")) {
    Write-Host "Generando datos de ejemplo..." -ForegroundColor Cyan
    python "$projectDir\generate_data.py"
}

# Ejecutar el proyecto según el modo especificado
Write-Host "Iniciando el proyecto en modo: $modo..." -ForegroundColor Green

switch ($modo) {
    "todo" {
        Write-Host "Ejecutando todos los componentes..." -ForegroundColor Yellow
          # Iniciar el nodo principal en una nueva ventana
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectDir'; python src/main.py --mode=train --head --config=$configPath" -WindowStyle Normal
        
        # Esperar un poco para que el nodo principal se inicie
        Start-Sleep -Seconds 10
        
        # Iniciar un nodo trabajador en una nueva ventana
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectDir'; python src/main.py --mode=train --worker --head-address=localhost:6379 --config=$configPath" -WindowStyle Normal
        
        # Iniciar el servicio de modelos en una nueva ventana
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectDir'; python src/main.py --mode=serve --config=$configPath" -WindowStyle Normal
        
        # Iniciar el monitoreo en la ventana actual
        python src/main.py --mode=monitor --config=$configPath
    }    "entrenamiento" {
        Write-Host "¿Desea iniciar un nodo principal (P) o un nodo trabajador (T)? [P/T]" -ForegroundColor Yellow
        $tipo = Read-Host
        
        if ($tipo -eq "P" -or $tipo -eq "p") {
            Write-Host "Iniciando nodo principal..." -ForegroundColor Cyan
            python src/main.py --mode=train --head --config=$configPath
        }
        elseif ($tipo -eq "T" -or $tipo -eq "t") {
            $dirIP = Read-Host "Ingrese la dirección IP del nodo principal (por defecto: localhost)"
            if (-not $dirIP) { $dirIP = "localhost" }
            
            Write-Host "Iniciando nodo trabajador conectado a $dirIP..." -ForegroundColor Cyan
            python src/main.py --mode=train --worker --head-address=$dirIP`:6379 --config=$configPath
        }
        else {
            Write-Host "Opción no válida. Iniciando nodo principal..." -ForegroundColor Cyan
            python src/main.py --mode=train --head --config=$configPath
        }
    }    "servicio" {
        Write-Host "Ejecutando servicio de modelos..." -ForegroundColor Yellow
        python src/main.py --mode=serve --config=$configPath
    }
    "monitoreo" {
        Write-Host "Ejecutando monitoreo..." -ForegroundColor Yellow
        python src/main.py --mode=monitor --config=$configPath
    }
    default {
        Write-Host "Modo no reconocido. Ejecutando en modo completo..." -ForegroundColor Yellow
        python src/main.py --mode=train --head --config=$configPath
    }
}