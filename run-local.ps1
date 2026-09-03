param([int]$Port = 8080)
Write-Host "Opening RoboPet at http://localhost:$Port"
python -m http.server $Port --directory (Join-Path $PSScriptRoot 'web')
