@echo off
FOR /F "tokens=* USEBACKQ" %%F IN (`cscript //nologo randomImage.vbs`) DO (
set imagePath=%%F
)
reg add "HKCU\Control Panel\Desktop" /v WallPaper /d %imagePath% /f
RUNDLL32.EXE USER32.DLL,UpdatePerUserSystemParameters ,1 ,True