param([int]$Port = 8080, [string]$Pi = 'ws://raspberrypi.local:8765', [switch]$Vlm)
Write-Host "Opening RoboPet control server at http://localhost:$Port"
if ($Vlm) {
    # Hybrid targeting: Qwen3-VL (llama-server) finds/verifies trash, CSRT tracks between answers.
    & "$PSScriptRoot\scripts\start-vlm-server.ps1"
    python -m autonomy.control_server --port $Port --pi $Pi --targeter vlm
} else {
    python -m autonomy.control_server --port $Port --pi $Pi
}
