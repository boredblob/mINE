Set WshShell = CreateObject("WScript.Shell") 
WshShell.Run "powershell -File C:\Users\omerk\Documents\GitHub\mINE\desktop\module.ps1 & exit", 0
Set WshShell = Nothing