# Query the ML Platform API
Write-Host "Querying the ML Platform API..." -ForegroundColor Green

# Health check
Write-Host "Checking API health..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
    Write-Host "API Health Status: $($health.status)" -ForegroundColor Green
    Write-Host "Models loaded: $($health.models_loaded)" -ForegroundColor Green
    Write-Host "Model names: $($health.model_names -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "Error connecting to API: $_" -ForegroundColor Red
}

# List available models
Write-Host "`nListing available models..." -ForegroundColor Cyan
try {
    $models = Invoke-RestMethod -Uri "http://localhost:8000/models" -Method Get
    Write-Host "Available models:" -ForegroundColor Green
    foreach ($model in $models.models) {
        Write-Host "- $model" -ForegroundColor White
    }
} catch {
    Write-Host "Error getting models: $_" -ForegroundColor Red
}

# Get metrics
Write-Host "`nGetting API metrics..." -ForegroundColor Cyan
try {
    $metrics = Invoke-RestMethod -Uri "http://localhost:8000/metrics" -Method Get
    Write-Host "Request counts:" -ForegroundColor Green
    $metrics.request_counts | ConvertTo-Json | Write-Host
    
    Write-Host "`nAverage latency (ms):" -ForegroundColor Green
    $metrics.average_latency_ms | ConvertTo-Json | Write-Host
} catch {
    Write-Host "Error getting metrics: $_" -ForegroundColor Red
}

Write-Host "`nTo make a prediction, use:" -ForegroundColor Cyan
Write-Host 'Invoke-RestMethod -Uri "http://localhost:8000/predict/MODEL_NAME" -Method Post -Body ''{"features":[{"feature_1":5.1,"feature_2":3.5,"feature_3":1.4,"feature_4":0.2}]}'' -ContentType "application/json"' -ForegroundColor Gray
