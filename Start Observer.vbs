Option Explicit

Dim shell, files, root, command, launcher
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

' Keep interpreter discovery in one visible/debuggable entry point.
launcher = files.BuildPath(root, "Start Observer.cmd")
command = "cmd.exe /d /s /c " & Chr(34) & Chr(34) & launcher & Chr(34) & Chr(34)
shell.Run command, 0, False
