Option Explicit

Dim shell, files, root, pythonw, condaPrefix, command
Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root

condaPrefix = shell.ExpandEnvironmentStrings("%CONDA_PREFIX%")
pythonw = ""

If files.FileExists("D:\450\conda\envs\tiktok\pythonw.exe") Then
  pythonw = "D:\450\conda\envs\tiktok\pythonw.exe"
End If

If pythonw = "" And condaPrefix <> "%CONDA_PREFIX%" Then
  If LCase(files.GetFileName(condaPrefix)) = "tiktok" Then
    If files.FileExists(files.BuildPath(condaPrefix, "pythonw.exe")) Then
      pythonw = files.BuildPath(condaPrefix, "pythonw.exe")
    End If
  End If
End If

If pythonw <> "" Then
  command = """" & pythonw & """ -m observer.launcher"
Else
  command = "cmd.exe /d /s /c ""conda run -n tiktok pythonw -m observer.launcher"""
End If

shell.Run command, 0, False
