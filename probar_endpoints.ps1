# Script para probar endpoints de la API de modelos
# filepath: d:\Escuela\Redes\Proyecto Distribuido\scr_pasd_2025\probar_endpoints.ps1

param(
    [Parameter()]
    [string]$host = "localhost",
    
    [Parameter()]
    [int]$port = 8000,
    
    [Parameter()]
    [ValidateSet('local', 'docker')]
    [string]$entorno = "local"
)

# Navegar al directorio del proyecto
$projectDir = $PSScriptRoot
Set-Location $projectDir

# Verificar si la API está en ejecución
function Test-ApiReady {
    try {
        Invoke-RestMethod -Uri "http://${host}:${port}" -Method GET -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

$baseUrl = "http://${host}:${port}"

Write-Host "=== Probando Endpoints de API ===" -ForegroundColor Magenta
Write-Host "Entorno: $entorno" -ForegroundColor Magenta
Write-Host "URL Base: $baseUrl" -ForegroundColor Cyan

# Verificar si la API está en ejecución
if (-not (Test-ApiReady)) {
    Write-Host "La API no parece estar disponible en $baseUrl" -ForegroundColor Yellow
    $iniciar = Read-Host "¿Desea iniciar el servicio de API? [S/N]"
    
    if ($iniciar -eq "S" -or $iniciar -eq "s") {
        if ($entorno -eq "docker") {
            Write-Host "Iniciando servicio en Docker..." -ForegroundColor Cyan
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectDir'; .\ejecutar_docker.ps1 -componente serving" -WindowStyle Normal
        }
        else {
            Write-Host "Iniciando servicio localmente..." -ForegroundColor Cyan
            Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$projectDir'; .\ejecutar_local.ps1 -modo servicio" -WindowStyle Normal
        }
        
        Write-Host "Esperando a que el servicio esté disponible..." -ForegroundColor Yellow
        $maxAttempts = 30
        $attempts = 0
        $ready = $false
        
        while (-not $ready -and $attempts -lt $maxAttempts) {
            $attempts++
            Start-Sleep -Seconds 2
            $ready = Test-ApiReady
            Write-Host "." -NoNewline
        }
        
        if ($ready) {
            Write-Host "`nServicio disponible!" -ForegroundColor Green
        }
        else {
            Write-Host "`nEl servicio no responde después de $($attempts*2) segundos. Compruebe los logs para ver si hay errores." -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "Continuando con las pruebas. Algunos endpoints pueden fallar si el servicio no está disponible." -ForegroundColor Yellow
    }
}

# Función para hacer peticiones HTTP y mostrar respuestas formateadas
function Invoke-ApiRequest {
    param(
        [string]$Endpoint,
        [string]$Method = "GET",
        [object]$Body = $null,
        [string]$Description = ""
    )
    
    $url = "${baseUrl}${Endpoint}"
    
    Write-Host "`n>> Probando: $Description" -ForegroundColor Yellow
    Write-Host "   $Method $url" -ForegroundColor Gray
    
    try {
        $params = @{
            Uri = $url
            Method = $Method
            ContentType = "application/json"
        }
        
        if ($Body -ne $null) {
            $jsonBody = $Body | ConvertTo-Json -Compress
            $params.Body = $jsonBody
            Write-Host "   Payload: $jsonBody" -ForegroundColor Gray
        }
        
        $response = Invoke-RestMethod @params -ErrorVariable responseError
        
        Write-Host "✓ Éxito!" -ForegroundColor Green
        Write-Host "   Respuesta:" -ForegroundColor Gray
        $response | ConvertTo-Json -Depth 4 | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
        return $response
    }
    catch {
        Write-Host "✗ Error ($($_.Exception.Response.StatusCode)):" -ForegroundColor Red
        $errorMsg = if ($responseError) { $responseError.Message } else { $_.Exception.Message }
        Write-Host "   $errorMsg" -ForegroundColor Red
        return $null
    }
}

# Probar endpoint de estado
Invoke-ApiRequest -Endpoint "/" -Description "Estado del servicio"

# Probar listado de modelos
$modelos = Invoke-ApiRequest -Endpoint "/models" -Description "Listar modelos disponibles"

if ($modelos) {
    # Si hay modelos, probar predicción con el primero de ellos
    if ($modelos.models.Count -gt 0) {
        $primerModelo = $modelos.models[0]
        Write-Host "`n>> Probando predicción con modelo: $primerModelo" -ForegroundColor Yellow
        
        # Datos de ejemplo para Iris (ajustar según el modelo)
        $datosPrueba = @{
            features = @(
                @(5.1, 3.5, 1.4, 0.2)  # Ejemplo con datos de iris
            )
        }
        
        # Hacer predicción
        Invoke-ApiRequest -Endpoint "/predict/$primerModelo" -Method "POST" -Body $datosPrueba -Description "Predicción con modelo $primerModelo"
    }
    else {
        Write-Host "`n✗ No hay modelos disponibles para probar predicciones" -ForegroundColor Yellow
    }
}

# Probar información del sistema
Invoke-ApiRequest -Endpoint "/system/info" -Description "Información del sistema"

Write-Host "`n=== Prueba de Endpoints Completada ===" -ForegroundColor Magenta
