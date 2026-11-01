# Upload the Piper dataset and start onnx training on a vast.ai box you already rented.
# Does not rent a machine and does not spend money.
$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
$cfg = @{}
Get-Content (Join-Path $PSScriptRoot 'config.env') | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq '' -or $line.StartsWith('#')) { return }
    $pair = $line.Split('=', 2)
    $cfg[$pair[0].Trim()] = $pair[1].Trim()
}
$hostName = $cfg['VAST_SSH_HOST']
$port = $cfg['VAST_SSH_PORT']
$user = $cfg['VAST_SSH_USER']
$remote = $cfg['VAST_REMOTE_DIR']
if (-not $hostName -or -not $port) {
    Write-Host "Fill VAST_SSH_HOST and VAST_SSH_PORT in vast/config.env."
    exit 1
}
$target = "${user}@${hostName}"
& ssh -p $port -o StrictHostKeyChecking=accept-new $target "mkdir -p '$remote'"
& scp -P $port -o StrictHostKeyChecking=accept-new -r `
    (Join-Path $Root 'datasets\piper') `
    (Join-Path $Root 'src') `
    (Join-Path $Root 'vast') `
    "${target}:${remote}/"
& ssh -p $port $target "mkdir -p '$remote/datasets/piper' && rm -rf '$remote/datasets/piper' && mv '$remote/piper' '$remote/datasets/piper' && chmod +x '$remote/vast/onstart_piper.sh' && bash '$remote/vast/onstart_piper.sh'"
