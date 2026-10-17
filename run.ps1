# Avvia il playground Laya.
#   .\run.ps1                    -> GPU (cuda), porta 8000
#   .\run.ps1 --device cpu       -> forza la CPU
#   .\run.ps1 --port 8080
param(
    [string]$Device = "",
    [string]$Port = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

uv sync --python 3.12

$args = @("run", "python", "src/server.py", "--warm")
if ($Device) { $args += @("--device", $Device) }
if ($Port)   { $args += @("--port", $Port) }

Write-Host "Avvio: uv $($args -join ' ')" -ForegroundColor Cyan
uv @args
