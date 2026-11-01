# Upload this project to a vast.ai instance you already rented, then start training.
# Fill vast/config.env first. This script does not rent a machine and does not spend money.
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
    Write-Host "Rent $($cfg['VAST_GPU']), image $($cfg['VAST_IMAGE']), disk $($cfg['VAST_DISK_GB']) GB."
    Write-Host "Copy the instance SSH host and port into that file, then run this script again."
    exit 1
}
$target = "${user}@${hostName}"
& ssh -p $port -o StrictHostKeyChecking=accept-new $target "mkdir -p '$remote'"
& scp -P $port -o StrictHostKeyChecking=accept-new -r `
    (Join-Path $Root 'datasets') `
    (Join-Path $Root 'src') `
    (Join-Path $Root 'vast') `
    (Join-Path $Root 'vendor') `
    (Join-Path $Root 'requirements-train.txt') `
    "${target}:${remote}/"
& ssh -p $port $target "chmod +x '$remote/vast/onstart.sh' && bash '$remote/vast/onstart.sh'"
