$folderName=(Get-ChildItem "C:\Program Files (x86)\Google\Chrome\Application\" -Name | Select-Object -First 1)
$VisualElementsText=$(@"
<Application xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <VisualElements
      ShowNameOnSquare150x150Logo='on'
      Square150x150Logo='$($foldername)\VisualElements\Logo.png'
      Square70x70Logo='$($foldername)\VisualElements\SmallLogo.png'
      Square44x44Logo='$($foldername)\VisualElements\SmallLogo.png'
      ForegroundText='light'/>
</Application>
"@)

Get-Variable -Name "VisualElementsText" -ValueOnly | Out-File -FilePath "C:\Program Files (x86)\Google\Chrome\Application\chrome.VisualElementsManifest.xml" -Encoding utf8 -NoNewline
Copy-Item -Path "C:\Users\omerk\Documents\Photoshop\Chrome Icons\*" -Destination "C:\Program Files (x86)\Google\Chrome\Application\$($folderName)\VisualElements" -Recurse
Rename-Item -Path "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk" -NewName "Google Chrome.lnk"

$wShell = New-Object -ComObject "wscript.shell"
$wShell.SendKeys("^{ESC}")
$wShell = ""

Rename-Item -Path "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Google Chromee.lnk" -NewName "Google Chrome.lnk"