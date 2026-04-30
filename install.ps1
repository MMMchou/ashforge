# Ashforge Installer for Windows
# Usage: irm https://raw.githubusercontent.com/MMMchou/ashforge/main/install.ps1 | iex

$ErrorActionPreference = "Stop"
$Repo = "MMMchou/ashforge"
$BinName = "ashforge.exe"
$InstallDir = "$env:USERPROFILE\.ashforge\bin"

Write-Host "Ashforge Installer" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host ""

# Detect architecture
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "386" }
Write-Host "Detected: windows/$Arch"

# Get latest release
Write-Host "Fetching latest release..."
try {
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
    $Tag = $Release.tag_name
} catch {
    Write-Host "Error: could not fetch latest release." -ForegroundColor Red
    Write-Host "Check https://github.com/$Repo/releases"
    exit 1
}
Write-Host "Latest version: $Tag"

# Build download URL
$Asset = "ashforge-windows-${Arch}.zip"
$Url = "https://github.com/$Repo/releases/download/$Tag/$Asset"

# Download
$TmpDir = Join-Path $env:TEMP "ashforge-install-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null
$ZipPath = Join-Path $TmpDir "ashforge.zip"

Write-Host "Downloading $Url..."
try {
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
} catch {
    # Fallback: try raw .exe
    $Url = "https://github.com/$Repo/releases/download/$Tag/ashforge.exe"
    Write-Host "Trying raw binary: $Url..."
    try {
        Invoke-WebRequest -Uri $Url -OutFile (Join-Path $TmpDir $BinName) -UseBasicParsing
    } catch {
        Write-Host "Error: download failed." -ForegroundColor Red
        Write-Host "Check available assets at: https://github.com/$Repo/releases/tag/$Tag"
        Remove-Item -Recurse -Force $TmpDir
        exit 1
    }
}

# Extract if zip
if (Test-Path $ZipPath) {
    Expand-Archive -Path $ZipPath -DestinationPath $TmpDir -Force
}

# Create install directory
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}

# Find and move all files (ashforge.exe + llama-server-cuda.exe + DLLs)
$ExePath = Get-ChildItem -Path $TmpDir -Filter $BinName -Recurse | Select-Object -First 1
if (-not $ExePath) {
    Write-Host "Error: $BinName not found in download." -ForegroundColor Red
    Remove-Item -Recurse -Force $TmpDir
    exit 1
}
# Copy all files from the same directory as ashforge.exe
$SrcDir = $ExePath.DirectoryName
Get-ChildItem -Path $SrcDir -File | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination (Join-Path $InstallDir $_.Name) -Force
}

# Clean up
Remove-Item -Recurse -Force $TmpDir

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host "Added $InstallDir to user PATH." -ForegroundColor Green
}

# Verify
Write-Host ""
$AshforgePath = Join-Path $InstallDir $BinName
if (Test-Path $AshforgePath) {
    Write-Host "Ashforge installed successfully!" -ForegroundColor Green
    Write-Host ""
    & $AshforgePath version
    Write-Host ""
    Write-Host "Get started:" -ForegroundColor Cyan
    Write-Host "  ashforge run Qwen3-30B-A3B"
    Write-Host ""
    Write-Host "Note: restart your terminal for PATH changes to take effect." -ForegroundColor Yellow
} else {
    Write-Host "Error: installation failed." -ForegroundColor Red
    exit 1
}
