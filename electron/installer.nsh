; NSIS custom install/uninstall hooks for OpenASL-Setup.exe (Zero-OBS Virtual Webcam)
!macro customInstall
  ExecWait 'regsvr32 /s "$INSTDIR\resources\driver\UnityCaptureFilter64.dll"'
!macroend

!macro customUnInstall
  ExecWait 'regsvr32 /u /s "$INSTDIR\resources\driver\UnityCaptureFilter64.dll"'
!macroend
