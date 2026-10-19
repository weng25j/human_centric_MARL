# Requires SUMO GUI and an activated Python environment.
param(
    [Parameter(Mandatory=$true)][string]$ShepherdModelDir,
    [Parameter(Mandatory=$true)][string]$FlatModelDir
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python -m recording.record_flat_marl --model-dir $FlatModelDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m recording.record_traffic_shepherd --model-dir $ShepherdModelDir
exit $LASTEXITCODE
