# Download the finished weights and drop them where Gothic's converter looks.
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
$local = Join-Path $Root 'weights\bezimienny'
New-Item -ItemType Directory -Force -Path $local | Out-Null
& scp -P $port -o StrictHostKeyChecking=accept-new "${user}@${hostName}:${remote}/weights/bezimienny/*.pth" $local
$game = 'C:\Users\mimi\gothic-bezimienny\rvc\assets\weights'
New-Item -ItemType Directory -Force -Path $game | Out-Null
Copy-Item (Join-Path $local 'bezimienny*.pth') $game -Force
Get-ChildItem $game -Filter 'bezimienny*.pth' | Select-Object Name, Length
