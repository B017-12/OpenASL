#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Registers the open-source UnityCapture DirectShow virtual webcam DLL on Windows.
  Zero OBS Studio or Unity installation required (~250 KB DLL).
#>

$ErrorActionPreference = "Stop"
$InstallDir = Join-Path $env:ProgramFiles "OpenASL\driver"
$DllPath = Join-Path $InstallDir "UnityCaptureFilter64.dll"
$DownloadUrl = "https://raw.githubusercontent.com/schellingb/UnityCapture/master/Install/UnityCaptureFilter64.dll"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if (-not (Test-Path $DllPath)) {
    Write-Host "[OpenASL] Downloading UnityCaptureFilter64.dll..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $DllPath -UseBasicParsing
}

Write-Host "[OpenASL] Registering OpenASL virtual webcam driver..."
Start-Process -FilePath "regsvr32.exe" -ArgumentList "/s", "`"$DllPath`"" -Wait -NoNewWindow

Write-Host "[OpenASL] Virtual camera registered! Select 'Unity Video Capture' in Zoom, Teams, Discord, or Omegle."
