Set fs = CreateObject("Scripting.FileSystemObject")

Set f = fs.GetFolder("C:\Users\omerk\Downloads\Desktop\slideshow\")
set lastFileNameFile = fs.openTextFile("./lastFileName.txt", 1)
lastFileName = lastFileNameFile.readAll()
lastFileNameFile.close()

differentNames = false
do while differentNames = false
  Randomize()
  i = Int((Rnd() * f.Files.Count) + 1)
  j = 1

  For Each fi In f.Files
    If j >= i Then
      if not lastFileName = fi.name then
        set lastFileNameFile = fs.openTextFile("./lastFileName.txt", 2)
        lastFileNameFile.write(fi.name)
        lastFileNameFile.close()
        differentNames = true
        WScript.echo(fi.path)
        Exit For
      end if
    End If
    j = j + 1
  Next
loop

lastFileNameFile.close()

WScript.quit(1)