Set WshShell = CreateObject("WScript.Shell") 
WshShell.Run "powershell -file ./run.ps1 & exit", 0
Set WshShell = Nothing