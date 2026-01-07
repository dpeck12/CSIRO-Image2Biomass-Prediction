# Downloads competition data into data/ using Kaggle CLI
# Requires kaggle.json configured (see setup_kaggle.ps1)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $repoRoot
$dataDir = Join-Path $repoRoot "data"
if (!(Test-Path $dataDir)) { New-Item -ItemType Directory -Path $dataDir | Out-Null }

Push-Location $dataDir

# Download and unzip
kaggle competitions download -c csiro-biomass

$zipFiles = Get-ChildItem -Filter "*.zip"
foreach ($z in $zipFiles) {
  Expand-Archive -Path $z.FullName -DestinationPath $dataDir -Force
}

Pop-Location
Write-Host "Downloaded and extracted competition data to $dataDir"
