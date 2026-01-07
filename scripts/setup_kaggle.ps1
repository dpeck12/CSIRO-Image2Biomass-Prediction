# Sets up Kaggle API credentials on Windows
# Place kaggle.json in %USERPROFILE%\.kaggle
# Get your token from: https://www.kaggle.com/settings

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
if (!(Test-Path $kaggleDir)) {
  New-Item -ItemType Directory -Path $kaggleDir | Out-Null
}

$src = Read-Host "Enter path to kaggle.json"
$dest = Join-Path $kaggleDir "kaggle.json"
Copy-Item -Path $src -Destination $dest -Force
Write-Host "Copied kaggle.json to $dest"

# Restrict permissions (recommended)
& icacls $dest /inheritance:r | Out-Null
& icacls $dest /grant:r "$($env:USERNAME):R" | Out-Null
Write-Host "Set permissions on kaggle.json"
