# Start llama-server with Qwen3-VL-4B (Q4_K_M GGUF) for --targeter vlm. No-op if it is already healthy.
# -Device picks the GPU (see `llama-server --list-devices`, e.g. Vulkan1 or CUDA0); empty = llama.cpp's default.
param([int]$Port = 8091, [string]$Device = '')
$health = "http://127.0.0.1:$Port/health"
try { if ((Invoke-RestMethod $health -TimeoutSec 2).status -eq 'ok') { Write-Host "VLM server already running on $Port"; return } } catch {}
$snap = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--Qwen--Qwen3-VL-4B-Instruct-GGUF\snapshots'
$model = Get-ChildItem $snap -Recurse -Filter 'Qwen3VL-4B-Instruct-Q4_K_M.gguf' | Select-Object -First 1
$mmproj = Get-ChildItem $snap -Recurse -Filter 'mmproj-Qwen3VL-4B-Instruct-F16.gguf' | Select-Object -First 1
if (-not $model -or -not $mmproj) { throw "Qwen3-VL GGUF not found under $snap (hf download Qwen/Qwen3-VL-4B-Instruct-GGUF)" }
$log = Join-Path (Split-Path $PSScriptRoot) 'llama-server.log'
$serverArgs = @('-m', "`"$($model.FullName)`"", '--mmproj', "`"$($mmproj.FullName)`"", '-ngl', '99',
    '-c', '4096', '--host', '127.0.0.1', '--port', $Port, '--no-webui')
if ($Device) { $serverArgs += @('--device', $Device) }
Start-Process llama-server -WindowStyle Hidden -RedirectStandardError $log -ArgumentList $serverArgs
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try { if ((Invoke-RestMethod $health -TimeoutSec 2).status -eq 'ok') { Write-Host "VLM server ready on $Port"; return } } catch {}
}
throw "llama-server did not become healthy; see $log"
