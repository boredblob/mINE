Get-ChildItem C:\Users\omerk\Downloads\Desktop\slideshow\ | Get-Random | %{.\setWallpaper.exe $_.FullName}