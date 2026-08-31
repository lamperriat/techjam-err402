[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight", "candidate")]
    [string]$Mode,

    [Parameter(Mandatory = $true)]
    [string]$ImplementationCommit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# This is the sole external formal orchestrator for SR-v2.21.  It intentionally
# has no data-set paths: all data/source gates remain inside the source-only
# Python runner.  The outer process owns the irreversible claim -> raw outer ->
# terminal transaction so a non-zero nested exit can never discard its receipt.

$ProjectRoot = "D:\tiktok\techjam-v2-21-gloss-g0"
$ProjectRootPosix = "D:/tiktok/techjam-v2-21-gloss-g0"
$RuntimeBase = "D:\tiktok\.v221_runtime"
$RuntimeBasePosix = "D:/tiktok/.v221_runtime"
$PythonExe = "D:\450\conda\envs\tiktok\python.exe"
$PythonBytes = 93184L
$PythonSha256 = "7819c841b9a6457da034e567563de1283dbc0b86482fd83d62b5d982d2a83a63"
$GitExe = "C:\Program Files\Git\mingw64\bin\git.exe"
$GitBytes = 4018680L
$GitSha256 = "3fe4878d8399f6fb7632b9325559d1bb38c3a17aac7a60f667c1e5f90b865248"
$GitVersion = "git version 2.45.2.windows.1"
$GitDir = "D:\tiktok\techjam-err402\.git\worktrees\techjam-v2-21-gloss-g0"
$GitDirPosix = "D:/tiktok/techjam-err402/.git/worktrees/techjam-v2-21-gloss-g0"
$Branch = "small-ranker-v2.21-gloss-g0"
$ParentCommit = "d7bc963188e1ba357539c22e75a016611ad52ba2"
$PreregCommit = "b81351a0657411ab04810bb4740b35b407d175cc"
$PreregBlob = "5ee89ebda59c3dbf973fc3cd3f127ec34f47d1fa"
$PreregBytes = 19911L
$PreregSha256 = "4f454f736d27762723e4b395036a518babed0299677510ddd35b03c62abdfb53"
$ProbeSchemaVersion = "small-ranker-v2.21-dual-view-rrf-g0-probe.v1"
$RemoteUrl = "https://github.com/lamperriat/techjam-err402.git"
$PreregRelative = "configs/small_ranker_v2_21.dual_view_rrf_g0_preregistration.json"
$BootstrapRelative = "scripts/v221_safe_bootstrap.py"
$RunnerRelative = "scripts/probe_sparse_union_g0.py"
$WorkerRelative = "scripts/sparse_union_g0_worker.py"
$UnionRelative = "starter/sparse_union_g0.py"
$PowerShellRelative = "scripts/run_v221_preflight.ps1"
$TestRelative = "tests/test_sparse_union_g0.py"
$ImplementationPaths = @(
    $RunnerRelative,
    $WorkerRelative,
    $BootstrapRelative,
    $PowerShellRelative,
    $UnionRelative,
    $TestRelative
) | Sort-Object
$TargetSourcePaths = @($BootstrapRelative, $RunnerRelative, $WorkerRelative)
$ZeroBlob = "0000000000000000000000000000000000000000"
$MaximumCaptureBytes = 1048576
$MaximumTerminalBytes = 1048576
$ProcessTimeoutMilliseconds = 1900000
$RecordedOn = "2026-08-31"

$FastTrack = Join-Path $ProjectRoot "experiments\fast_track"
$PreregPath = Join-Path $ProjectRoot ($PreregRelative -replace "/", "\")
$BootstrapPath = Join-Path $ProjectRoot ($BootstrapRelative -replace "/", "\")
$RunnerPath = Join-Path $ProjectRoot ($RunnerRelative -replace "/", "\")
$WorkerPath = Join-Path $ProjectRoot ($WorkerRelative -replace "/", "\")
$WorktreeDotGit = Join-Path $ProjectRoot ".git"

$ModePaths = @{
    preflight = [ordered]@{
        claim = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_preflight_claim_20260831.json"
        outer = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_preflight_outer_20260831.json"
        result = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_preflight_20260831.json"
    }
    candidate = [ordered]@{
        claim = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_claim_20260831.json"
        outer = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_outer_20260831.json"
        result = Join-Path $FastTrack "small_ranker_v2_21_dual_view_rrf_g0_candidate_recall_20260831.json"
    }
}

function Throw-Code {
    param([Parameter(Mandatory = $true)][string]$Code)
    throw [System.InvalidOperationException]::new($Code)
}

function Get-SafeErrorCode {
    param([Parameter(Mandatory = $true)]$Caught)
    try {
        $message = [string]$Caught.Exception.Message
        if ($message -cmatch "^[A-Z][A-Z0-9_]{1,79}$") {
            return $message
        }
    }
    catch {
        # Deliberately discard every raw exception string.
    }
    return "UNEXPECTED_ORCHESTRATOR_FAILURE"
}

function Get-FullPathKey {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path).TrimEnd("\").ToUpperInvariant()
}

function Assert-PlainExistingPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet("File", "Directory")][string]$Kind
    )
    $full = [System.IO.Path]::GetFullPath($Path)
    $leaf = Get-Item -LiteralPath $full -Force -ErrorAction Stop
    if ($Kind -eq "File" -and -not ($leaf -is [System.IO.FileInfo])) {
        Throw-Code "PATH_TYPE_DRIFT"
    }
    if ($Kind -eq "Directory" -and -not ($leaf -is [System.IO.DirectoryInfo])) {
        Throw-Code "PATH_TYPE_DRIFT"
    }
    $current = $leaf
    while ($null -ne $current) {
        if (($current.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            Throw-Code "REPARSE_PATH_DENIED"
        }
        $parentText = [System.IO.Path]::GetDirectoryName($current.FullName)
        if ([string]::IsNullOrEmpty($parentText) -or (Get-FullPathKey $parentText) -eq (Get-FullPathKey $current.FullName)) {
            break
        }
        $current = Get-Item -LiteralPath $parentText -Force -ErrorAction Stop
    }
    if ((Get-FullPathKey $leaf.FullName) -ne (Get-FullPathKey $full)) {
        Throw-Code "PATH_ALIAS_DENIED"
    }
    return $leaf
}

function Test-LiteralPathExists {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.File]::Exists($Path) -or [System.IO.Directory]::Exists($Path)
}

function Get-PlainFileBytes {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [int64]$MaximumBytes = 4194304L
    )
    $item = Assert-PlainExistingPath -Path $Path -Kind File
    if ($item.Length -lt 0 -or $item.Length -gt $MaximumBytes) {
        Throw-Code "FILE_SIZE_GATE"
    }
    $beforeLength = [int64]$item.Length
    $beforeWrite = $item.LastWriteTimeUtc.Ticks
    $bytes = [System.IO.File]::ReadAllBytes($item.FullName)
    $after = Assert-PlainExistingPath -Path $Path -Kind File
    if ($bytes.LongLength -ne $beforeLength -or [int64]$after.Length -ne $beforeLength -or $after.LastWriteTimeUtc.Ticks -ne $beforeWrite) {
        Throw-Code "FILE_MUTATION"
    }
    return ,$bytes
}

function Get-Sha256Hex {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-GitBlobHex {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)
    $header = [System.Text.Encoding]::ASCII.GetBytes("blob $($Bytes.LongLength)`0")
    $payload = New-Object byte[] ($header.Length + $Bytes.Length)
    [System.Buffer]::BlockCopy($header, 0, $payload, 0, $header.Length)
    [System.Buffer]::BlockCopy($Bytes, 0, $payload, $header.Length, $Bytes.Length)
    $algorithm = [System.Security.Cryptography.SHA1]::Create()
    try {
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($payload))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Convert-CrlfToLf {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)
    $stream = New-Object System.IO.MemoryStream
    try {
        for ($index = 0; $index -lt $Bytes.Length; $index++) {
            if ($Bytes[$index] -eq 13 -and ($index + 1) -lt $Bytes.Length -and $Bytes[$index + 1] -eq 10) {
                continue
            }
            $stream.WriteByte($Bytes[$index])
        }
        return ,$stream.ToArray()
    }
    finally {
        $stream.Dispose()
    }
}

function Assert-WorktreeBlob {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedBlob
    )
    if ($ExpectedBlob -cnotmatch "^[0-9a-f]{40}$") {
        Throw-Code "GIT_BLOB_SHAPE"
    }
    $raw = Get-PlainFileBytes -Path $Path -MaximumBytes 8388608
    $normalized = Convert-CrlfToLf -Bytes $raw
    $normalizedBlob = Get-GitBlobHex -Bytes $normalized
    # Git-LF bytes are authoritative for all six frozen text files.  Accepting
    # a raw CRLF blob whose normalized object differs would only defer a known
    # fail-closed condition to the already-consumed Python process.
    if ($normalizedBlob -cne $ExpectedBlob) {
        Throw-Code "WORKTREE_BLOB_DRIFT"
    }
}

function Quote-NativeArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.IndexOf([char]0) -ge 0 -or $Value -match '[\r\n"]') {
        Throw-Code "NATIVE_ARGUMENT_DENIED"
    }
    if ($Value -match "\s") {
        # All dynamic formal arguments are frozen, no-space paths or hex.  A
        # whitespace-bearing argument would therefore be an implementation bug.
        Throw-Code "NATIVE_ARGUMENT_WHITESPACE"
    }
    return $Value
}

function Invoke-CapturedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$Environment,
        [Parameter(Mandatory = $true)][int]$TimeoutMilliseconds
    )
    Assert-PlainExistingPath -Path $FileName -Kind File | Out-Null
    Assert-PlainExistingPath -Path $WorkingDirectory -Kind Directory | Out-Null
    $argumentText = (($Arguments | ForEach-Object { Quote-NativeArgument ([string]$_) }) -join " ")
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $FileName
    $start.Arguments = $argumentText
    $start.WorkingDirectory = $WorkingDirectory
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.EnvironmentVariables.Clear()
    foreach ($entry in $Environment.GetEnumerator()) {
        $start.EnvironmentVariables[[string]$entry.Key] = [string]$entry.Value
    }
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    $stdout = New-Object System.IO.MemoryStream
    $stderr = New-Object System.IO.MemoryStream
    $timedOut = $false
    try {
        if (-not $process.Start()) {
            Throw-Code "PROCESS_START_FAILED"
        }
        $stdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        $stderrTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            $timedOut = $true
            try { $process.Kill() } catch { }
            try { $process.WaitForExit() } catch { }
        }
        if (-not $stdoutTask.Wait(60000) -or -not $stderrTask.Wait(60000)) {
            Throw-Code "PROCESS_CAPTURE_TIMEOUT"
        }
        $exitCode = if ($timedOut) { 2 } else { [int]$process.ExitCode }
        return [pscustomobject]@{
            ExitCode = $exitCode
            Stdout = $stdout.ToArray()
            Stderr = $stderr.ToArray()
            TimedOut = $timedOut
        }
    }
    finally {
        $stdout.Dispose()
        $stderr.Dispose()
        $process.Dispose()
    }
}

function New-MinimalEnvironment {
    param([string]$TempPath = "")
    $environment = [ordered]@{}
    foreach ($name in @("COMSPEC", "PATHEXT", "SYSTEMDRIVE", "SYSTEMROOT", "WINDIR")) {
        $value = [System.Environment]::GetEnvironmentVariable($name)
        if (-not [string]::IsNullOrEmpty($value)) {
            $environment[$name] = $value
        }
    }
    $systemRoot = [System.Environment]::GetEnvironmentVariable("SYSTEMROOT")
    if ([string]::IsNullOrEmpty($systemRoot)) {
        $systemRoot = "C:\Windows"
        $environment["SYSTEMROOT"] = $systemRoot
        $environment["WINDIR"] = $systemRoot
    }
    $environment["PATH"] = Join-Path $systemRoot "System32"
    if (-not [string]::IsNullOrEmpty($TempPath)) {
        $environment["TEMP"] = $TempPath
        $environment["TMP"] = $TempPath
    }
    return $environment
}

function New-GitEnvironment {
    $environment = New-MinimalEnvironment
    $environment["GIT_ATTR_NOSYSTEM"] = "1"
    $environment["GIT_CONFIG_GLOBAL"] = "NUL"
    $environment["GIT_CONFIG_NOSYSTEM"] = "1"
    $environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    $environment["GIT_OPTIONAL_LOCKS"] = "0"
    $environment["GIT_PAGER"] = "cat"
    $environment["GIT_TERMINAL_PROMPT"] = "0"
    $environment["LANG"] = "C"
    $environment["LC_ALL"] = "C"
    $environment["PAGER"] = "cat"
    return $environment
}

function Convert-Utf8Strict {
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes)
    $encoding = New-Object System.Text.UTF8Encoding($false, $true)
    try {
        return $encoding.GetString($Bytes)
    }
    catch {
        Throw-Code "UTF8_CONTRACT"
    }
}

function Invoke-FrozenGit {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [int[]]$AllowedExitCodes = @(0)
    )
    $prefix = @(
        "--git-dir=$GitDirPosix",
        "--work-tree=$ProjectRootPosix",
        "--no-replace-objects"
    )
    $captured = Invoke-CapturedProcess -FileName $GitExe -Arguments ($prefix + $Arguments) -WorkingDirectory "C:\Windows\System32" -Environment (New-GitEnvironment) -TimeoutMilliseconds 30000
    if ($captured.TimedOut -or $captured.Stdout.Length -gt 1048576 -or $captured.Stderr.Length -ne 0 -or $AllowedExitCodes -notcontains $captured.ExitCode) {
        Throw-Code "GIT_COMMAND_FAILED"
    }
    $text = Convert-Utf8Strict -Bytes $captured.Stdout
    return [pscustomobject]@{ ExitCode = $captured.ExitCode; Text = $text }
}

function Get-GitSingleLine {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $result = Invoke-FrozenGit -Arguments $Arguments
    $value = $result.Text.TrimEnd("`r", "`n")
    if ([string]::IsNullOrEmpty($value) -or $value.Contains("`r") -or $value.Contains("`n")) {
        Throw-Code "GIT_SINGLE_LINE_CONTRACT"
    }
    return $value
}

function Assert-SmallControlFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedNormalized
    )
    $raw = Get-PlainFileBytes -Path $Path -MaximumBytes 4096
    $text = Convert-Utf8Strict -Bytes $raw
    $normalized = $text.Replace("`r`n", "`n")
    if ($normalized -cne $ExpectedNormalized) {
        Throw-Code "GIT_CONTROL_PLANE_DRIFT"
    }
}

function Assert-ExecutableFingerprint {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int64]$ExpectedBytes,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    $raw = Get-PlainFileBytes -Path $Path -MaximumBytes 8388608
    if ($raw.LongLength -ne $ExpectedBytes -or (Get-Sha256Hex -Bytes $raw) -cne $ExpectedSha256) {
        Throw-Code "EXECUTABLE_FINGERPRINT_DRIFT"
    }
}

function Assert-GitControlPlane {
    Assert-PlainExistingPath -Path $ProjectRoot -Kind Directory | Out-Null
    Assert-PlainExistingPath -Path $GitDir -Kind Directory | Out-Null
    Assert-SmallControlFile -Path $WorktreeDotGit -ExpectedNormalized "gitdir: $GitDirPosix`n"
    Assert-SmallControlFile -Path (Join-Path $GitDir "HEAD") -ExpectedNormalized "ref: refs/heads/$Branch`n"
    Assert-SmallControlFile -Path (Join-Path $GitDir "commondir") -ExpectedNormalized "../..`n"
    Assert-SmallControlFile -Path (Join-Path $GitDir "gitdir") -ExpectedNormalized "$ProjectRootPosix/.git`n"
    foreach ($forbidden in @(
        (Join-Path $GitDir "shallow"),
        (Join-Path $GitDir "config.worktree"),
        "D:\tiktok\techjam-err402\.git\shallow",
        "D:\tiktok\techjam-err402\.git\objects\info\alternates"
    )) {
        if (Test-LiteralPathExists $forbidden) {
            Throw-Code "GIT_CONTROL_PLANE_UNSUPPORTED"
        }
    }
    Assert-ExecutableFingerprint -Path $GitExe -ExpectedBytes $GitBytes -ExpectedSha256 $GitSha256
    $version = Get-GitSingleLine -Arguments @("--version")
    if ($version -cne $GitVersion) {
        Throw-Code "GIT_VERSION_DRIFT"
    }
}

function Get-CommitBlob {
    param(
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$RelativePath
    )
    $value = Get-GitSingleLine -Arguments @("rev-parse", "--verify", "$Commit`:$RelativePath")
    if ($value -cnotmatch "^[0-9a-f]{40}$") {
        Throw-Code "GIT_BLOB_SHAPE"
    }
    $kind = Get-GitSingleLine -Arguments @("cat-file", "-t", $value)
    if ($kind -cne "blob") {
        Throw-Code "GIT_OBJECT_TYPE"
    }
    return $value
}

function Assert-PushedCheckpoint {
    param([Parameter(Mandatory = $true)][string]$Commit)
    if ($Commit -cnotmatch "^[0-9a-f]{40}$") {
        Throw-Code "IMPLEMENTATION_COMMIT_SHAPE"
    }
    Assert-GitControlPlane
    Assert-ExecutableFingerprint -Path $PythonExe -ExpectedBytes $PythonBytes -ExpectedSha256 $PythonSha256

    $resolved = Get-GitSingleLine -Arguments @("rev-parse", "--verify", "$Commit^{commit}")
    $local = Get-GitSingleLine -Arguments @("rev-parse", "--verify", "refs/heads/$Branch^{commit}")
    $remote = Get-GitSingleLine -Arguments @("rev-parse", "--verify", "refs/remotes/origin/$Branch^{commit}")
    if ($resolved -cne $Commit -or $local -cne $Commit -or $remote -cne $Commit) {
        Throw-Code "PUSHED_REF_MISMATCH"
    }
    if ($Commit -ceq $PreregCommit) { Throw-Code "IMPLEMENTATION_COMMIT_REQUIRED" }
    $preregParent = Get-GitSingleLine -Arguments @("rev-parse", "$PreregCommit^")
    if ($preregParent -cne $ParentCommit) { Throw-Code "PREREG_PARENT_DRIFT" }
    $ancestor = Invoke-FrozenGit -Arguments @("merge-base", "--is-ancestor", $PreregCommit, $Commit) -AllowedExitCodes @(0, 1)
    if ($ancestor.ExitCode -ne 0) {
        Throw-Code "PREREG_NOT_ANCESTOR"
    }
    $mergeCommits = (Invoke-FrozenGit -Arguments @("rev-list", "--min-parents=2", "$PreregCommit..$Commit")).Text.Trim()
    if (-not [string]::IsNullOrEmpty($mergeCommits)) { Throw-Code "IMPLEMENTATION_MERGE_DENIED" }
    $changedRaw = (Invoke-FrozenGit -Arguments @("diff-tree", "--no-commit-id", "--name-only", "--no-renames", "-r", $PreregCommit, $Commit)).Text
    $changed = @($changedRaw -split "`r?`n" | Where-Object { -not [string]::IsNullOrEmpty($_) } | Sort-Object)
    if ($changed.Count -ne $ImplementationPaths.Count) {
        Throw-Code "IMPLEMENTATION_PATH_SET_DRIFT"
    }
    for ($index = 0; $index -lt $ImplementationPaths.Count; $index++) {
        if ($changed[$index] -cne $ImplementationPaths[$index]) {
            Throw-Code "IMPLEMENTATION_PATH_SET_DRIFT"
        }
    }

    $preregAtFrozen = Get-CommitBlob -Commit $PreregCommit -RelativePath $PreregRelative
    $preregAtImplementation = Get-CommitBlob -Commit $Commit -RelativePath $PreregRelative
    if ($preregAtFrozen -cne $PreregBlob -or $preregAtImplementation -cne $PreregBlob) {
        Throw-Code "PREREG_BLOB_DRIFT"
    }
    Assert-WorktreeBlob -Path $PreregPath -ExpectedBlob $PreregBlob

    $blobs = [ordered]@{}
    foreach ($relative in $ImplementationPaths) {
        $blob = Get-CommitBlob -Commit $Commit -RelativePath $relative
        $path = Join-Path $ProjectRoot ($relative -replace "/", "\")
        Assert-WorktreeBlob -Path $path -ExpectedBlob $blob
        $blobs[$relative] = $blob
    }
    foreach ($relative in $TargetSourcePaths) {
        $path = Join-Path $ProjectRoot ($relative -replace "/", "\")
        $raw = Get-PlainFileBytes -Path $path -MaximumBytes 8388608
        $text = Convert-Utf8Strict -Bytes $raw
        if ($text.Contains($ZeroBlob)) {
            Throw-Code "UNPATCHED_SOURCE_BLOB"
        }
        if ([string]$blobs[$relative] -cnotmatch "^[0-9a-f]{40}$") {
            Throw-Code "TARGET_SOURCE_BLOB_DRIFT"
        }
    }
    # This check deliberately happens before the immutable claim.  The
    # source-only bootstrap is the executable manifest for all three semantic
    # targets; a merely non-zero but stale hash must never consume the one-shot.
    $bootstrapRaw = Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608
    $bootstrapText = Convert-Utf8Strict -Bytes $bootstrapRaw
    foreach ($relative in @($RunnerRelative, $WorkerRelative, $UnionRelative)) {
        $expected = [string]$blobs[$relative]
        if ($expected -cnotmatch "^[0-9a-f]{40}$" -or -not $bootstrapText.Contains($expected)) {
            Throw-Code "BOOTSTRAP_MANIFEST_BLOB_DRIFT"
        }
    }
    return $blobs
}

function Escape-JsonString {
    param([AllowEmptyString()][string]$Value)
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    foreach ($character in $Value.ToCharArray()) {
        $code = [int][char]$character
        $escaped = $null
        switch ($code) {
            8 { $escaped = '\b' }
            9 { $escaped = '\t' }
            10 { $escaped = '\n' }
            12 { $escaped = '\f' }
            13 { $escaped = '\r' }
            34 { $escaped = '\"' }
            92 { $escaped = '\\' }
        }
        if ($null -ne $escaped) {
            [void]$builder.Append($escaped)
            continue
        }
        if ($code -lt 32) {
            [void]$builder.Append(("\u{0:x4}" -f $code))
        }
        else {
            [void]$builder.Append($character)
        }
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function ConvertTo-CanonicalJson {
    param($Value, [int]$Depth = 0)
    if ($Depth -gt 32) {
        Throw-Code "JSON_DEPTH_GATE"
    }
    if ($null -eq $Value) { return "null" }
    if ($Value -is [string] -or $Value -is [char]) { return Escape-JsonString ([string]$Value) }
    if ($Value -is [bool]) { if ($Value) { return "true" } else { return "false" } }
    if ($Value -is [byte] -or $Value -is [sbyte] -or $Value -is [int16] -or $Value -is [uint16] -or $Value -is [int32] -or $Value -is [uint32] -or $Value -is [int64] -or $Value -is [uint64]) {
        return ([System.Convert]::ToString($Value, [System.Globalization.CultureInfo]::InvariantCulture))
    }
    if ($Value -is [single] -or $Value -is [double]) {
        $number = [double]$Value
        if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { Throw-Code "JSON_NONFINITE" }
        return $number.ToString("R", [System.Globalization.CultureInfo]::InvariantCulture).Replace("E", "e")
    }
    if ($Value -is [decimal]) {
        return ([decimal]$Value).ToString([System.Globalization.CultureInfo]::InvariantCulture)
    }
    if ($Value -is [System.Collections.IDictionary]) {
        $keys = @($Value.Keys | ForEach-Object { [string]$_ } | Sort-Object)
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($key in $keys) {
            $parts.Add((Escape-JsonString $key) + ":" + (ConvertTo-CanonicalJson -Value $Value[$key] -Depth ($Depth + 1)))
        }
        return "{" + ($parts -join ",") + "}"
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $properties = @($Value.PSObject.Properties | Sort-Object Name)
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($property in $properties) {
            $parts.Add((Escape-JsonString $property.Name) + ":" + (ConvertTo-CanonicalJson -Value $property.Value -Depth ($Depth + 1)))
        }
        return "{" + ($parts -join ",") + "}"
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($entry in $Value) {
            $parts.Add((ConvertTo-CanonicalJson -Value $entry -Depth ($Depth + 1)))
        }
        return "[" + ($parts -join ",") + "]"
    }
    Throw-Code "JSON_TYPE_DENIED"
}

function Get-CanonicalBytes {
    param([Parameter(Mandatory = $true)]$Value)
    $json = ConvertTo-CanonicalJson -Value $Value
    return ,[System.Text.Encoding]::UTF8.GetBytes($json + "`n")
}

function Write-ExclusiveBytes {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][byte[]]$Bytes
    )
    $parent = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($Path))
    Assert-PlainExistingPath -Path $parent -Kind Directory | Out-Null
    $stream = [System.IO.FileStream]::new(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None,
        4096,
        [System.IO.FileOptions]::WriteThrough
    )
    try {
        $stream.Write($Bytes, 0, $Bytes.Length)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }
    $observed = Get-PlainFileBytes -Path $Path -MaximumBytes ([Math]::Max($Bytes.Length, 1))
    if ($observed.Length -ne $Bytes.Length -or (Get-Sha256Hex -Bytes $observed) -cne (Get-Sha256Hex -Bytes $Bytes)) {
        Throw-Code "DURABLE_WRITE_VERIFICATION"
    }
}

function Assert-ReceiptPrivacy {
    param($Value, [int]$Depth = 0)
    if ($Depth -gt 24) { Throw-Code "RECEIPT_DEPTH_GATE" }
    if ($null -eq $Value -or $Value -is [bool] -or $Value -is [ValueType]) { return }
    if ($Value -is [string]) {
        if ($Value.Length -gt 4096 -or $Value -cmatch "^B0[A-Z0-9]{8}$") { Throw-Code "RECEIPT_PRIVACY_GATE" }
        return
    }
    if ($Value -is [System.Collections.IDictionary]) {
        if ($Value.Count -gt 256) { Throw-Code "RECEIPT_COMPACTNESS_GATE" }
        foreach ($key in $Value.Keys) {
            $name = [string]$key
            if ($name -cmatch "^(asin|parent_asin|session_id|message|messages|query|queries|candidate_ids|identifiers|membership_vector|per_session)$") {
                Throw-Code "RECEIPT_PRIVACY_GATE"
            }
            Assert-ReceiptPrivacy -Value $Value[$key] -Depth ($Depth + 1)
        }
        return
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $properties = @($Value.PSObject.Properties)
        if ($properties.Count -gt 256) { Throw-Code "RECEIPT_COMPACTNESS_GATE" }
        foreach ($property in $properties) {
            if ($property.Name -cmatch "^(asin|parent_asin|session_id|message|messages|query|queries|candidate_ids|identifiers|membership_vector|per_session)$") {
                Throw-Code "RECEIPT_PRIVACY_GATE"
            }
            Assert-ReceiptPrivacy -Value $property.Value -Depth ($Depth + 1)
        }
        return
    }
    if ($Value -is [System.Collections.IEnumerable]) {
        $entries = @($Value)
        if ($entries.Count -gt 100) { Throw-Code "RECEIPT_ARRAY_GATE" }
        foreach ($entry in $entries) { Assert-ReceiptPrivacy -Value $entry -Depth ($Depth + 1) }
        return
    }
    Throw-Code "RECEIPT_TYPE_GATE"
}

function Test-BasicOuterEnvelope {
    param(
        [Parameter(Mandatory = $true)][byte[]]$Stdout,
        [Parameter(Mandatory = $true)][byte[]]$Stderr,
        [Parameter(Mandatory = $true)][bool]$TimedOut
    )
    if ($TimedOut -or $Stderr.Length -ne 0 -or $Stdout.Length -lt 32 -or $Stdout.Length -gt $MaximumCaptureBytes) { return $false }
    if ($Stdout[$Stdout.Length - 1] -ne 10) { return $false }
    for ($index = 0; $index -lt ($Stdout.Length - 1); $index++) {
        if ($Stdout[$index] -eq 10 -or $Stdout[$index] -eq 13 -or $Stdout[$index] -eq 0) { return $false }
    }
    try { $text = Convert-Utf8Strict -Bytes $Stdout } catch { return $false }
    return $text.StartsWith('{"bootstrap":', [System.StringComparison]::Ordinal) -and $text.EndsWith("}`n", [System.StringComparison]::Ordinal)
}

function Get-ExactPropertyNames {
    param([Parameter(Mandatory = $true)]$Value)
    if (-not ($Value -is [System.Management.Automation.PSCustomObject])) { Throw-Code "OUTER_JSON_TYPE" }
    return @($Value.PSObject.Properties.Name | Sort-Object)
}

function Assert-ExactNames {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string[]]$Expected
    )
    $actual = Get-ExactPropertyNames -Value $Value
    $wanted = @($Expected | Sort-Object)
    if ($actual.Count -ne $wanted.Count) { Throw-Code "OUTER_JSON_SCHEMA" }
    for ($index = 0; $index -lt $wanted.Count; $index++) {
        if ($actual[$index] -cne $wanted[$index]) { Throw-Code "OUTER_JSON_SCHEMA" }
    }
}

function Parse-And-ValidateOuter {
    param(
        [Parameter(Mandatory = $true)][byte[]]$Raw,
        [Parameter(Mandatory = $true)][string]$ExpectedRunnerBlob,
        [Parameter(Mandatory = $true)][string]$ExpectedBootstrapBlob,
        [Parameter(Mandatory = $true)][string]$ExpectedPycache
    )
    $text = Convert-Utf8Strict -Bytes $Raw
    try { $value = $text.TrimEnd("`n") | ConvertFrom-Json -ErrorAction Stop } catch { Throw-Code "OUTER_JSON_PARSE" }
    Assert-ExactNames -Value $value -Expected @("bootstrap", "target_exit_code", "target_receipt")
    Assert-ExactNames -Value $value.bootstrap -Expected @("bootstrap_blob", "guarded_path", "mode", "pycache_prefix", "source_only", "target_blob")
    if (
        [string]$value.bootstrap.bootstrap_blob -cne $ExpectedBootstrapBlob -or
        [string]$value.bootstrap.target_blob -cne $ExpectedRunnerBlob -or
        [string]$value.bootstrap.mode -cne "direct" -or
        $value.bootstrap.guarded_path -isnot [bool] -or $value.bootstrap.guarded_path -ne $true -or
        $value.bootstrap.source_only -isnot [bool] -or $value.bootstrap.source_only -ne $true -or
        (Get-FullPathKey ([string]$value.bootstrap.pycache_prefix)) -cne (Get-FullPathKey $ExpectedPycache)
    ) { Throw-Code "BOOTSTRAP_ATTESTATION_DRIFT" }
    if (-not ($value.target_exit_code -is [int] -or $value.target_exit_code -is [long]) -or [int64]$value.target_exit_code -lt 0 -or [int64]$value.target_exit_code -gt 255) {
        Throw-Code "TARGET_EXIT_CODE_SCHEMA"
    }
    if ($null -ne $value.target_receipt) { Assert-ReceiptPrivacy -Value $value.target_receipt }
    return $value
}

function New-SanitizedOuterFailure {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [int]$ProcessExitCode = 2,
        [int]$StdoutBytes = 0,
        [int]$StderrBytes = 0
    )
    return [ordered]@{
        capture_status = "NO_VALID_CANONICAL_ENVELOPE"
        error_code = $Code
        process_exit_code = $ProcessExitCode
        raw_stderr_retained = $false
        raw_stdout_retained = $false
        schema_version = "small-ranker-v2.21-outer-capture-failure.v1"
        stderr_bytes = $StderrBytes
        stdout_bytes = $StdoutBytes
    }
}

function Write-ConsoleCanonical {
    param([Parameter(Mandatory = $true)][byte[]]$Bytes)
    $stream = [System.Console]::OpenStandardOutput()
    $stream.Write($Bytes, 0, $Bytes.Length)
    $stream.Flush()
}

function Assert-FixedAttemptPathsAbsent {
    param([Parameter(Mandatory = $true)]$Paths)
    Assert-PlainExistingPath -Path $FastTrack -Kind Directory | Out-Null
    foreach ($name in @("claim", "outer", "result")) {
        $path = [string]$Paths[$name]
        if ((Get-FullPathKey ([System.IO.Path]::GetDirectoryName($path))) -cne (Get-FullPathKey $FastTrack)) {
            Throw-Code "ATTEMPT_PATH_DRIFT"
        }
        if (Test-LiteralPathExists $path) { Throw-Code "ONE_SHOT_PATH_PREEXISTS" }
    }
}

function ConvertFrom-CanonicalJsonBytes {
    param(
        [Parameter(Mandatory = $true)][byte[]]$Raw,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $text = Convert-Utf8Strict -Bytes $Raw
    if (-not $text.EndsWith("`n", [System.StringComparison]::Ordinal)) { Throw-Code $Code }
    try { $value = $text.TrimEnd("`n") | ConvertFrom-Json -ErrorAction Stop } catch { Throw-Code $Code }
    if (-not ($value -is [System.Management.Automation.PSCustomObject])) { Throw-Code $Code }
    $canonical = Get-CanonicalBytes -Value $value
    if ($canonical.Length -ne $Raw.Length -or (Get-Sha256Hex -Bytes $canonical) -cne (Get-Sha256Hex -Bytes $Raw)) {
        Throw-Code $Code
    }
    return $value
}

function ConvertFrom-PythonCanonicalOuterBytes {
    param([Parameter(Mandatory = $true)][byte[]]$Raw)
    $empty = New-Object byte[] 0
    if (-not (Test-BasicOuterEnvelope -Stdout $Raw -Stderr $empty -TimedOut $false)) {
        Throw-Code "PREFLIGHT_OUTER_SCHEMA"
    }
    $text = Convert-Utf8Strict -Bytes $Raw
    try { $value = $text.TrimEnd("`n") | ConvertFrom-Json -ErrorAction Stop } catch { Throw-Code "PREFLIGHT_OUTER_SCHEMA" }
    if (-not ($value -is [System.Management.Automation.PSCustomObject])) { Throw-Code "PREFLIGHT_OUTER_SCHEMA" }
    # Python's canonical encoder preserves a few floating-point spellings (for
    # example 0.0) that Windows PowerShell's round-trip serializer normalizes.
    # The source-only Python preclaim check below independently enforces exact
    # canonical bytes before the candidate claim is allowed to exist.
    return $value
}

function Get-RawIdentity {
    param([Parameter(Mandatory = $true)][byte[]]$Raw)
    return [ordered]@{ bytes = $Raw.Length; sha256 = (Get-Sha256Hex -Bytes $Raw) }
}

function Assert-IdentityShape {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Code
    )
    try { Assert-ExactNames -Value $Value -Expected @("bytes", "sha256") } catch { Throw-Code $Code }
    if (
        -not ($Value.bytes -is [int] -or $Value.bytes -is [long]) -or
        [int64]$Value.bytes -le 0 -or
        [string]$Value.sha256 -cnotmatch "^[0-9a-f]{64}$"
    ) { Throw-Code $Code }
}

function Assert-IdentityEqual {
    param(
        [Parameter(Mandatory = $true)]$Observed,
        [Parameter(Mandatory = $true)]$Expected,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-IdentityShape -Value $Observed -Code $Code
    if ([int64]$Observed.bytes -ne [int64]$Expected.bytes -or [string]$Observed.sha256 -cne [string]$Expected.sha256) {
        Throw-Code $Code
    }
}

function Assert-ExactBoolean {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][bool]$Expected,
        [Parameter(Mandatory = $true)][string]$Code
    )
    if ($Value -isnot [bool] -or $Value -ne $Expected) { Throw-Code $Code }
}

function Assert-ExactInteger {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][int64]$Expected,
        [Parameter(Mandatory = $true)][string]$Code
    )
    if (-not ($Value -is [int] -or $Value -is [long]) -or [int64]$Value -ne $Expected) {
        Throw-Code $Code
    }
}

function Assert-CanonicalObjectEqual {
    param(
        [Parameter(Mandatory = $true)]$Left,
        [Parameter(Mandatory = $true)]$Right,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $leftRaw = Get-CanonicalBytes -Value $Left
    $rightRaw = Get-CanonicalBytes -Value $Right
    if ($leftRaw.Length -ne $rightRaw.Length -or (Get-Sha256Hex -Bytes $leftRaw) -cne (Get-Sha256Hex -Bytes $rightRaw)) {
        Throw-Code $Code
    }
}

function Assert-PriorPycachePrefix {
    param([Parameter(Mandatory = $true)]$Value)
    if (-not ($Value -is [string])) { Throw-Code "PREFLIGHT_BOOTSTRAP_ATTESTATION" }
    try { $full = [System.IO.Path]::GetFullPath([string]$Value) } catch { Throw-Code "PREFLIGHT_BOOTSTRAP_ATTESTATION" }
    $runtimeRoot = [System.IO.Path]::GetDirectoryName($full)
    if (
        [System.IO.Path]::GetFileName($full) -cne "pycache" -or
        [System.IO.Path]::GetFileName($runtimeRoot) -cnotmatch "^v221-[0-9a-f]{32}$" -or
        (Get-FullPathKey ([System.IO.Path]::GetDirectoryName($runtimeRoot))) -cne (Get-FullPathKey $RuntimeBase)
    ) { Throw-Code "PREFLIGHT_BOOTSTRAP_ATTESTATION" }
}

function Assert-CandidatePrerequisite {
    param(
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)]$Blobs
    )
    if ($Mode -ne "candidate") { return $null }
    $preflight = $ModePaths["preflight"]
    foreach ($name in @("claim", "outer", "result")) {
        $path = [string]$preflight[$name]
        if (-not [System.IO.File]::Exists($path)) { Throw-Code "PREFLIGHT_TERMINAL_REQUIRED" }
        Assert-PlainExistingPath -Path $path -Kind File | Out-Null
    }

    $claimRaw = Get-PlainFileBytes -Path ([string]$preflight["claim"]) -MaximumBytes 65536
    $claim = ConvertFrom-CanonicalJsonBytes -Raw $claimRaw -Code "PREFLIGHT_CLAIM_SCHEMA"
    Assert-ExactNames -Value $claim -Expected @(
        "attempt_consumed", "branch", "experiment_id", "implementation_commit", "mode",
        "one_shot", "preregistration", "preregistration_commit", "recorded_on",
        "schema_version", "target_source_blobs"
    )
    Assert-ExactNames -Value $claim.preregistration -Expected @("blob", "commit")
    Assert-ExactNames -Value $claim.target_source_blobs -Expected @("bootstrap", "runner", "worker")
    if (
        [string]$claim.schema_version -cne "small-ranker-v2.21-durable-one-shot-claim.v1" -or
        [string]$claim.experiment_id -cne "SR-V2.21-CLAUSE-ISOLATED-DUAL-VIEW-RRF-G0" -or
        [string]$claim.branch -cne $Branch -or [string]$claim.mode -cne "preflight" -or
        [string]$claim.implementation_commit -cne $Commit -or $claim.one_shot -isnot [bool] -or $claim.one_shot -ne $true -or
        $claim.attempt_consumed -isnot [bool] -or $claim.attempt_consumed -ne $true -or
        [string]$claim.preregistration_commit -cne $PreregCommit -or
        [string]$claim.preregistration.commit -cne $PreregCommit -or [string]$claim.preregistration.blob -cne $PreregBlob -or
        [string]$claim.recorded_on -cne $RecordedOn -or
        [string]$claim.target_source_blobs.bootstrap -cne [string]$Blobs[$BootstrapRelative] -or
        [string]$claim.target_source_blobs.runner -cne [string]$Blobs[$RunnerRelative] -or
        [string]$claim.target_source_blobs.worker -cne [string]$Blobs[$WorkerRelative]
    ) { Throw-Code "PREFLIGHT_CLAIM_SEMANTICS" }
    $claimIdentity = Get-RawIdentity -Raw $claimRaw

    $outerRaw = Get-PlainFileBytes -Path ([string]$preflight["outer"]) -MaximumBytes $MaximumCaptureBytes
    $outer = ConvertFrom-PythonCanonicalOuterBytes -Raw $outerRaw
    Assert-ExactNames -Value $outer -Expected @("bootstrap", "target_exit_code", "target_receipt")
    Assert-ExactNames -Value $outer.bootstrap -Expected @("bootstrap_blob", "guarded_path", "mode", "pycache_prefix", "source_only", "target_blob")
    if (
        [string]$outer.bootstrap.bootstrap_blob -cne [string]$Blobs[$BootstrapRelative] -or
        [string]$outer.bootstrap.target_blob -cne [string]$Blobs[$RunnerRelative] -or
        [string]$outer.bootstrap.mode -cne "direct" -or
        $outer.bootstrap.guarded_path -isnot [bool] -or $outer.bootstrap.guarded_path -ne $true -or
        $outer.bootstrap.source_only -isnot [bool] -or $outer.bootstrap.source_only -ne $true -or
        -not ($outer.target_exit_code -is [int] -or $outer.target_exit_code -is [long]) -or
        [int64]$outer.target_exit_code -ne 0 -or
        -not ($outer.target_receipt -is [System.Management.Automation.PSCustomObject])
    ) { Throw-Code "PREFLIGHT_OUTER_NOT_COMPLETE" }
    Assert-PriorPycachePrefix -Value $outer.bootstrap.pycache_prefix
    $outerIdentity = Get-RawIdentity -Raw $outerRaw
    $targetReceipt = $outer.target_receipt
    Assert-ExactNames -Value $targetReceipt -Expected @(
        "bootstrap", "claim", "device", "entrypoint_regression", "experiment_id", "git",
        "implementation", "integrity", "mode", "next", "preregistration", "recorded_on",
        "rerun_forbidden", "runtime", "schema_version", "sources", "stages", "status"
    )
    Assert-IdentityEqual -Observed $targetReceipt.claim -Expected $claimIdentity -Code "PREFLIGHT_RECEIPT_CLAIM_BINDING"
    Assert-CanonicalObjectEqual -Left $targetReceipt.bootstrap -Right $outer.bootstrap -Code "PREFLIGHT_RECEIPT_BOOTSTRAP_BINDING"
    Assert-ExactNames -Value $targetReceipt.implementation -Expected @("branch", "commit", "default_off", "preregistration_commit", "served_top10_unchanged", "target_blind")
    Assert-ExactNames -Value $targetReceipt.integrity -Expected @("exact_triplet_each_stage", "legacy_route_executions", "network_attempt_count", "ordered_variable_c200_prefix", "target_sources_opened")
    Assert-ExactNames -Value $targetReceipt.device -Expected @("gpu_peak_bytes", "reason", "selected")
    Assert-ExactNames -Value $targetReceipt.entrypoint_regression -Expected @("legacy_module_denied_direct", "legacy_module_denied_module", "runner_direct", "runner_module", "worker_direct", "worker_module")
    Assert-ExactNames -Value $targetReceipt.sources -Expected @("catalog", "sealed_c200", "visible_context")
    Assert-ExactNames -Value $targetReceipt.git -Expected @("branch", "commit", "implementation_blobs", "object_only_git", "preregistration_commit", "pushed", "remote")
    Assert-ExactNames -Value $targetReceipt.git.implementation_blobs -Expected $ImplementationPaths
    Assert-ExactNames -Value $targetReceipt.preregistration -Expected @("bytes", "rows", "sha256")
    Assert-ExactBoolean -Value $targetReceipt.rerun_forbidden -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    foreach ($name in @("default_off", "target_blind", "served_top10_unchanged")) {
        Assert-ExactBoolean -Value $targetReceipt.implementation.$name -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    }
    Assert-ExactBoolean -Value $targetReceipt.integrity.exact_triplet_each_stage -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactBoolean -Value $targetReceipt.integrity.ordered_variable_c200_prefix -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactBoolean -Value $targetReceipt.integrity.target_sources_opened -Expected $false -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactInteger -Value $targetReceipt.integrity.legacy_route_executions -Expected 0 -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactInteger -Value $targetReceipt.integrity.network_attempt_count -Expected 0 -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactInteger -Value $targetReceipt.device.gpu_peak_bytes -Expected 0 -Code "PREFLIGHT_RECEIPT_TYPE"
    foreach ($name in @("runner_direct", "runner_module", "worker_direct", "worker_module", "legacy_module_denied_direct", "legacy_module_denied_module")) {
        Assert-ExactBoolean -Value $targetReceipt.entrypoint_regression.$name -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    }
    Assert-ExactBoolean -Value $targetReceipt.git.pushed -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactBoolean -Value $targetReceipt.git.object_only_git -Expected $true -Code "PREFLIGHT_RECEIPT_TYPE"
    Assert-ExactInteger -Value $targetReceipt.preregistration.bytes -Expected $PreregBytes -Code "PREFLIGHT_RECEIPT_PREREGISTRATION"
    if (
        -not ($targetReceipt.preregistration.rows -is [int] -or $targetReceipt.preregistration.rows -is [long]) -or
        [int64]$targetReceipt.preregistration.rows -le 0 -or
        [string]$targetReceipt.preregistration.sha256 -cne $PreregSha256
    ) { Throw-Code "PREFLIGHT_RECEIPT_PREREGISTRATION" }
    $stages = @($targetReceipt.stages)
    if (
        [string]$targetReceipt.schema_version -cne $ProbeSchemaVersion -or
        [string]$targetReceipt.experiment_id -cne "SR-V2.21-CLAUSE-ISOLATED-DUAL-VIEW-RRF-G0" -or
        [string]$targetReceipt.mode -cne "preflight" -or [string]$targetReceipt.status -cne "TARGET_FREE_PREFLIGHT_COMPLETE" -or
        [string]$targetReceipt.recorded_on -cne $RecordedOn -or $targetReceipt.rerun_forbidden -isnot [bool] -or $targetReceipt.rerun_forbidden -ne $true -or
        [string]$targetReceipt.implementation.branch -cne $Branch -or [string]$targetReceipt.implementation.commit -cne $Commit -or
        [string]$targetReceipt.implementation.preregistration_commit -cne $PreregCommit -or
        $targetReceipt.implementation.default_off -ne $true -or $targetReceipt.implementation.target_blind -ne $true -or $targetReceipt.implementation.served_top10_unchanged -ne $true -or
        $targetReceipt.integrity.exact_triplet_each_stage -ne $true -or $targetReceipt.integrity.ordered_variable_c200_prefix -ne $true -or
        [int64]$targetReceipt.integrity.legacy_route_executions -ne 0 -or $targetReceipt.integrity.target_sources_opened -ne $false -or [int64]$targetReceipt.integrity.network_attempt_count -ne 0 -or
        [string]$targetReceipt.device.selected -cne "CPU" -or [string]$targetReceipt.device.reason -cne "frozen sparse FTS/mask/Fraction-RRF backend" -or [int64]$targetReceipt.device.gpu_peak_bytes -ne 0 -or
        $targetReceipt.entrypoint_regression.runner_direct -ne $true -or $targetReceipt.entrypoint_regression.runner_module -ne $true -or
        $targetReceipt.entrypoint_regression.worker_direct -ne $true -or $targetReceipt.entrypoint_regression.worker_module -ne $true -or
        $targetReceipt.entrypoint_regression.legacy_module_denied_direct -ne $true -or $targetReceipt.entrypoint_regression.legacy_module_denied_module -ne $true -or
        [string]$targetReceipt.git.branch -cne $Branch -or [string]$targetReceipt.git.commit -cne $Commit -or
        [string]$targetReceipt.git.preregistration_commit -cne $PreregCommit -or [string]$targetReceipt.git.remote -cne $RemoteUrl -or
        $targetReceipt.git.pushed -ne $true -or $targetReceipt.git.object_only_git -ne $true -or $stages.Count -ne 2
    ) { Throw-Code "PREFLIGHT_RECEIPT_NOT_ELIGIBLE" }
    foreach ($relative in $ImplementationPaths) {
        if ([string]$targetReceipt.git.implementation_blobs.PSObject.Properties[$relative].Value -cne [string]$Blobs[$relative]) {
            Throw-Code "PREFLIGHT_RECEIPT_SOURCE_BLOBS"
        }
    }
    for ($index = 0; $index -lt 2; $index++) {
        $expectedLimit = @(20, 100)[$index]
        if (-not ($stages[$index].session_limit -is [int] -or $stages[$index].session_limit -is [long]) -or [int64]$stages[$index].session_limit -ne $expectedLimit -or $stages[$index].exact_triplet -isnot [bool] -or $stages[$index].exact_triplet -ne $true) {
            Throw-Code "PREFLIGHT_STAGE_GATE"
        }
    }

    $terminalRaw = Get-PlainFileBytes -Path ([string]$preflight["result"]) -MaximumBytes $MaximumTerminalBytes
    $terminal = ConvertFrom-CanonicalJsonBytes -Raw $terminalRaw -Code "PREFLIGHT_TERMINAL_SCHEMA"
    Assert-ExactNames -Value $terminal -Expected @(
        "implementation_commit", "mode", "outer", "preregistration", "process_exit_code",
        "raw_stderr_retained", "recorded_on", "schema_version", "status",
        "target_exit_code", "target_receipt"
    )
    Assert-ExactNames -Value $terminal.preregistration -Expected @("blob", "commit")
    Assert-IdentityEqual -Observed $terminal.outer -Expected $outerIdentity -Code "PREFLIGHT_TERMINAL_OUTER_BINDING"
    Assert-CanonicalObjectEqual -Left $terminal.target_receipt -Right $targetReceipt -Code "PREFLIGHT_TERMINAL_RECEIPT_BINDING"
    if (
        [string]$terminal.schema_version -cne "small-ranker-v2.21-durable-terminal.v1" -or
        [string]$terminal.status -cne "COMPLETE" -or [string]$terminal.mode -cne "preflight" -or
        [string]$terminal.implementation_commit -cne $Commit -or [string]$terminal.recorded_on -cne $RecordedOn -or
        -not ($terminal.process_exit_code -is [int] -or $terminal.process_exit_code -is [long]) -or [int64]$terminal.process_exit_code -ne 0 -or
        -not ($terminal.target_exit_code -is [int] -or $terminal.target_exit_code -is [long]) -or [int64]$terminal.target_exit_code -ne 0 -or
        $terminal.raw_stderr_retained -isnot [bool] -or $terminal.raw_stderr_retained -ne $false -or
        [string]$terminal.preregistration.commit -cne $PreregCommit -or [string]$terminal.preregistration.blob -cne $PreregBlob
    ) { Throw-Code "PREFLIGHT_NOT_COMPLETE" }
    Assert-ReceiptPrivacy -Value $terminal
    $terminalIdentity = Get-RawIdentity -Raw $terminalRaw
    return [ordered]@{ claim = $claimIdentity; outer = $outerIdentity; terminal = $terminalIdentity }
}

function New-FreshRuntime {
    Assert-PlainExistingPath -Path "D:\tiktok" -Kind Directory | Out-Null
    if (-not [System.IO.Directory]::Exists($RuntimeBase)) {
        [void][System.IO.Directory]::CreateDirectory($RuntimeBase)
    }
    Assert-PlainExistingPath -Path $RuntimeBase -Kind Directory | Out-Null
    $nonce = [System.Guid]::NewGuid().ToString("N")
    $runtimeRoot = Join-Path $RuntimeBase ("v221-" + $nonce)
    if (Test-LiteralPathExists $runtimeRoot) { Throw-Code "RUNTIME_COLLISION" }
    [void][System.IO.Directory]::CreateDirectory($runtimeRoot)
    Assert-PlainExistingPath -Path $runtimeRoot -Kind Directory | Out-Null
    if ((Get-FullPathKey ([System.IO.Path]::GetDirectoryName($runtimeRoot))) -cne (Get-FullPathKey $RuntimeBase)) { Throw-Code "RUNTIME_PARENT_DRIFT" }
    $pycache = Join-Path $runtimeRoot "pycache"
    $temp = Join-Path $runtimeRoot "temp"
    [void][System.IO.Directory]::CreateDirectory($pycache)
    [void][System.IO.Directory]::CreateDirectory($temp)
    Assert-PlainExistingPath -Path $pycache -Kind Directory | Out-Null
    Assert-PlainExistingPath -Path $temp -Kind Directory | Out-Null
    if ((Get-ChildItem -LiteralPath $pycache -Force | Measure-Object).Count -ne 0 -or (Get-ChildItem -LiteralPath $temp -Force | Measure-Object).Count -ne 0) {
        Throw-Code "RUNTIME_NOT_FRESH"
    }
    return [pscustomobject]@{ Root = $runtimeRoot; Pycache = $pycache; Temp = $temp }
}

function New-OfflineEnvironment {
    param([Parameter(Mandatory = $true)][string]$TempPath)
    $environment = New-MinimalEnvironment -TempPath $TempPath
    $environment["CUDA_VISIBLE_DEVICES"] = ""
    $environment["HF_HUB_OFFLINE"] = "1"
    $environment["MKL_NUM_THREADS"] = "1"
    $environment["OMP_NUM_THREADS"] = "1"
    $environment["OPENBLAS_NUM_THREADS"] = "1"
    $environment["PYTHONDONTWRITEBYTECODE"] = "1"
    $environment["PYTHONHASHSEED"] = "0"
    $environment["PYTHONNOUSERSITE"] = "1"
    $environment["TOKENIZERS_PARALLELISM"] = "false"
    return $environment
}

function Remove-OwnedFreshRuntime {
    param([Parameter(Mandatory = $true)]$Runtime)
    $root = [string]$Runtime.Root
    Assert-PlainExistingPath -Path $RuntimeBase -Kind Directory | Out-Null
    Assert-PlainExistingPath -Path $root -Kind Directory | Out-Null
    if (
        (Get-FullPathKey ([System.IO.Path]::GetDirectoryName($root))) -cne (Get-FullPathKey $RuntimeBase) -or
        (Get-FullPathKey $root) -ceq (Get-FullPathKey $RuntimeBase)
    ) { Throw-Code "RUNTIME_CLEANUP_SCOPE" }
    $children = @(Get-ChildItem -LiteralPath $root -Force | Sort-Object Name)
    $allowed = @("module", "pycache", "temp")
    foreach ($child in $children) {
        if ($allowed -cnotcontains $child.Name) { Throw-Code "RUNTIME_CLEANUP_CONTENT" }
        if (($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { Throw-Code "RUNTIME_CLEANUP_REPARSE" }
    }
    foreach ($emptyName in @("pycache", "temp")) {
        $emptyPath = Join-Path $root $emptyName
        Assert-PlainExistingPath -Path $emptyPath -Kind Directory | Out-Null
        if ((Get-ChildItem -LiteralPath $emptyPath -Force | Measure-Object).Count -ne 0) { Throw-Code "RUNTIME_CLEANUP_CONTENT" }
    }
    $moduleRoot = Join-Path $root "module"
    if ([System.IO.Directory]::Exists($moduleRoot)) {
        Assert-PlainExistingPath -Path $moduleRoot -Kind Directory | Out-Null
        $moduleChildren = @(Get-ChildItem -LiteralPath $moduleRoot -Force)
        if ($moduleChildren.Count -ne 1 -or $moduleChildren[0].Name -cne "v221_safe_bootstrap.py" -or -not ($moduleChildren[0] -is [System.IO.FileInfo])) {
            Throw-Code "RUNTIME_CLEANUP_CONTENT"
        }
        Assert-PlainExistingPath -Path $moduleChildren[0].FullName -Kind File | Out-Null
    }
    [System.IO.Directory]::Delete($root, $true)
    if (Test-LiteralPathExists $root) { Throw-Code "RUNTIME_CLEANUP_FAILED" }
}

function Invoke-PreclaimEntrypointCheck {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct", "module")][string]$InvocationMode,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [Parameter(Mandatory = $true)][string]$TargetModule,
        [Parameter(Mandatory = $true)][string]$TargetBlob,
        [Parameter(Mandatory = $true)][string]$BootstrapBlob
    )
    $checkRuntime = $null
    try {
        $checkRuntime = New-FreshRuntime
        $pycachePosix = $checkRuntime.Pycache.Replace("\", "/")
        $environment = New-OfflineEnvironment -TempPath $checkRuntime.Temp
        $arguments = @("-P", "-S", "-s", "-B", "-X", "pycache_prefix=$pycachePosix")
        if ($InvocationMode -eq "direct") {
            $arguments += $BootstrapPath.Replace("\", "/")
        }
        else {
            $moduleRoot = Join-Path $checkRuntime.Root "module"
            [void][System.IO.Directory]::CreateDirectory($moduleRoot)
            Assert-PlainExistingPath -Path $moduleRoot -Kind Directory | Out-Null
            $modulePath = Join-Path $moduleRoot "v221_safe_bootstrap.py"
            $bootstrapBytes = Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608
            Write-ExclusiveBytes -Path $modulePath -Bytes $bootstrapBytes
            $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
            $arguments += @("-m", "v221_safe_bootstrap")
        }
        $arguments += @(
            "--mode", $InvocationMode,
            "--target-path", $TargetPath.Replace("\", "/"),
            "--target-module", $TargetModule,
            "--target-blob", $TargetBlob,
            "--bootstrap-blob", $BootstrapBlob,
            "--",
            "--entrypoint-self-check", "--require-module", "starter.sparse_union_g0"
        )
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_ENTRYPOINT_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob $TargetBlob -ExpectedBootstrapBlob $BootstrapBlob -ExpectedPycache $checkRuntime.Pycache
        if ($capture.ExitCode -ne 0 -or [int]$outer.target_exit_code -ne 0 -or $null -eq $outer.target_receipt) {
            Throw-Code "PRECLAIM_ENTRYPOINT_EXIT"
        }
        Assert-ExactNames -Value $outer.target_receipt -Expected @(
            "c200_contract_imported", "evaluator_imported", "legacy_runtime_absent",
            "project_root_bootstrapped", "required_module", "status"
        )
        if (
            $outer.target_receipt.c200_contract_imported -isnot [bool] -or $outer.target_receipt.c200_contract_imported -ne $true -or
            $outer.target_receipt.evaluator_imported -isnot [bool] -or $outer.target_receipt.evaluator_imported -ne $true -or
            $outer.target_receipt.legacy_runtime_absent -isnot [bool] -or $outer.target_receipt.legacy_runtime_absent -ne $true -or
            $outer.target_receipt.project_root_bootstrapped -isnot [bool] -or $outer.target_receipt.project_root_bootstrapped -ne $true -or
            [string]$outer.target_receipt.required_module -cne "starter.sparse_union_g0" -or
            [string]$outer.target_receipt.status -cne "ENTRYPOINT_SELF_CHECK_PASS"
        ) { Throw-Code "PRECLAIM_ENTRYPOINT_RECEIPT" }
    }
    finally {
        if ($null -ne $checkRuntime -and [System.IO.Directory]::Exists([string]$checkRuntime.Root)) {
            Remove-OwnedFreshRuntime -Runtime $checkRuntime
        }
    }
}

function Invoke-PreclaimEntrypointChecks {
    param([Parameter(Mandatory = $true)]$Blobs)
    $targets = @(
        [pscustomobject]@{ Path = $RunnerPath; Module = "scripts.probe_sparse_union_g0"; Relative = $RunnerRelative },
        [pscustomobject]@{ Path = $WorkerPath; Module = "scripts.sparse_union_g0_worker"; Relative = $WorkerRelative }
    )
    foreach ($target in $targets) {
        foreach ($invocationMode in @("direct", "module")) {
            Invoke-PreclaimEntrypointCheck -InvocationMode $invocationMode -TargetPath $target.Path -TargetModule $target.Module -TargetBlob ([string]$Blobs[$target.Relative]) -BootstrapBlob ([string]$Blobs[$BootstrapRelative])
        }
    }
}

function Invoke-PreclaimPrerequisiteCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)]$Blobs,
        [Parameter(Mandatory = $true)]$ExpectedPrerequisite
    )
    if ($Mode -ne "candidate") { return }
    $checkRuntime = $null
    try {
        $checkRuntime = New-FreshRuntime
        $pycachePosix = $checkRuntime.Pycache.Replace("\", "/")
        $arguments = @(
            "-P", "-S", "-s", "-B", "-X", "pycache_prefix=$pycachePosix",
            $BootstrapPath.Replace("\", "/"),
            "--mode", "direct",
            "--target-path", $RunnerPath.Replace("\", "/"),
            "--target-module", "scripts.probe_sparse_union_g0",
            "--target-blob", [string]$Blobs[$RunnerRelative],
            "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
            "--",
            "--preclaim-chain-self-check", "--implementation-commit", $Commit
        )
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment (New-OfflineEnvironment -TempPath $checkRuntime.Temp) -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_PREREQUISITE_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedPycache $checkRuntime.Pycache
        if ($capture.ExitCode -ne 0 -or [int]$outer.target_exit_code -ne 0 -or $null -eq $outer.target_receipt) {
            Throw-Code "PRECLAIM_PREREQUISITE_EXIT"
        }
        Assert-ExactNames -Value $outer.target_receipt -Expected @("implementation_commit", "preflight_prerequisite", "status")
        if (
            [string]$outer.target_receipt.status -cne "PRECLAIM_PREFLIGHT_CHAIN_PASS" -or
            [string]$outer.target_receipt.implementation_commit -cne $Commit
        ) { Throw-Code "PRECLAIM_PREREQUISITE_RECEIPT" }
        Assert-ExactNames -Value $outer.target_receipt.preflight_prerequisite -Expected @("claim", "outer", "terminal")
        foreach ($name in @("claim", "outer", "terminal")) {
            Assert-IdentityShape -Value $outer.target_receipt.preflight_prerequisite.$name -Code "PRECLAIM_PREREQUISITE_IDENTITY"
        }
        Assert-CanonicalObjectEqual -Left $outer.target_receipt.preflight_prerequisite -Right $ExpectedPrerequisite -Code "PRECLAIM_PREREQUISITE_DIVERGENCE"
    }
    finally {
        if ($null -ne $checkRuntime -and [System.IO.Directory]::Exists([string]$checkRuntime.Root)) {
            Remove-OwnedFreshRuntime -Runtime $checkRuntime
        }
    }
}

$claimed = $false
$outerWritten = $false
$terminalWritten = $false
$attemptPaths = $null
$runtime = $null
$sourceBlobs = $null
$preflightPrerequisite = $null
$processCapture = $null
$outerRaw = $null
$outerIsRawEnvelope = $false
$parsedOuter = $null
$finalBytes = $null
$finalExitCode = 2

try {
    $Mode = $Mode.ToLowerInvariant()
    if (@("preflight", "candidate") -cnotcontains $Mode) { Throw-Code "MODE_SHAPE" }
    $ImplementationCommit = $ImplementationCommit.ToLowerInvariant()
    if ($ImplementationCommit -cnotmatch "^[0-9a-f]{40}$") { Throw-Code "IMPLEMENTATION_COMMIT_SHAPE" }
    if ((Get-FullPathKey (Get-Location).Path) -cne (Get-FullPathKey $ProjectRoot)) { Throw-Code "FORMAL_CWD_DRIFT" }
    $attemptPaths = $ModePaths[$Mode]
    $sourceBlobs = Assert-PushedCheckpoint -Commit $ImplementationCommit
    $preflightPrerequisite = Assert-CandidatePrerequisite -Commit $ImplementationCommit -Blobs $sourceBlobs
    Assert-FixedAttemptPathsAbsent -Paths $attemptPaths
    Invoke-PreclaimEntrypointChecks -Blobs $sourceBlobs
    $finalSourceBlobs = Assert-PushedCheckpoint -Commit $ImplementationCommit
    Assert-CanonicalObjectEqual -Left $finalSourceBlobs -Right $sourceBlobs -Code "PRECLAIM_SOURCE_MUTATION"
    if ($Mode -eq "candidate") {
        $finalPrerequisite = Assert-CandidatePrerequisite -Commit $ImplementationCommit -Blobs $sourceBlobs
        Assert-CanonicalObjectEqual -Left $finalPrerequisite -Right $preflightPrerequisite -Code "PRECLAIM_PREREQUISITE_MUTATION"
        Invoke-PreclaimPrerequisiteCheck -Commit $ImplementationCommit -Blobs $sourceBlobs -ExpectedPrerequisite $preflightPrerequisite
    }
    Assert-FixedAttemptPathsAbsent -Paths $attemptPaths

    $claim = [ordered]@{
        attempt_consumed = $true
        branch = $Branch
        experiment_id = "SR-V2.21-CLAUSE-ISOLATED-DUAL-VIEW-RRF-G0"
        implementation_commit = $ImplementationCommit
        mode = $Mode
        one_shot = $true
        preregistration = [ordered]@{ blob = $PreregBlob; commit = $PreregCommit }
        preregistration_commit = $PreregCommit
        recorded_on = $RecordedOn
        schema_version = "small-ranker-v2.21-durable-one-shot-claim.v1"
        target_source_blobs = [ordered]@{
            bootstrap = [string]$sourceBlobs[$BootstrapRelative]
            runner = [string]$sourceBlobs[$RunnerRelative]
            worker = [string]$sourceBlobs[$WorkerRelative]
        }
    }
    if ($Mode -eq "candidate") {
        $claim["preflight_prerequisite"] = $preflightPrerequisite
    }
    $claimBytes = Get-CanonicalBytes -Value $claim
    Write-ExclusiveBytes -Path ([string]$attemptPaths["claim"]) -Bytes $claimBytes
    $claimed = $true

    try {
        $runtime = New-FreshRuntime
        $pycachePosix = $runtime.Pycache.Replace("\", "/")
        $arguments = @(
            "-P", "-S", "-s", "-B", "-X", "pycache_prefix=$pycachePosix",
            $BootstrapPath.Replace("\", "/"),
            "--mode", "direct",
            "--target-path", $RunnerPath.Replace("\", "/"),
            "--target-module", "scripts.probe_sparse_union_g0",
            "--target-blob", [string]$sourceBlobs[$RunnerRelative],
            "--bootstrap-blob", [string]$sourceBlobs[$BootstrapRelative],
            "--",
            "--run", "--mode", $Mode, "--implementation-commit", $ImplementationCommit
        )
        $processCapture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment (New-OfflineEnvironment -TempPath $runtime.Temp) -TimeoutMilliseconds $ProcessTimeoutMilliseconds
        if ($processCapture.Stdout.Length -gt $MaximumCaptureBytes -or $processCapture.Stderr.Length -gt $MaximumCaptureBytes) {
            Throw-Code "OUTER_CAPTURE_SIZE_GATE"
        }
        $outerIsRawEnvelope = Test-BasicOuterEnvelope -Stdout $processCapture.Stdout -Stderr $processCapture.Stderr -TimedOut $processCapture.TimedOut
        if ($outerIsRawEnvelope) {
            $outerRaw = $processCapture.Stdout
        }
        else {
            $failureCode = if ($processCapture.TimedOut) { "FORMAL_PROCESS_TIMEOUT" } elseif ($processCapture.Stderr.Length -ne 0) { "FORMAL_STDERR_NONEMPTY" } else { "NO_CANONICAL_OUTER_ENVELOPE" }
            $outerRaw = Get-CanonicalBytes -Value (New-SanitizedOuterFailure -Code $failureCode -ProcessExitCode $processCapture.ExitCode -StdoutBytes $processCapture.Stdout.Length -StderrBytes $processCapture.Stderr.Length)
        }
        Write-ExclusiveBytes -Path ([string]$attemptPaths["outer"]) -Bytes $outerRaw
        $outerWritten = $true

        if ($outerIsRawEnvelope) {
            $parsedOuter = Parse-And-ValidateOuter -Raw $outerRaw -ExpectedRunnerBlob ([string]$sourceBlobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$sourceBlobs[$BootstrapRelative]) -ExpectedPycache $runtime.Pycache
        }
        $complete = $outerIsRawEnvelope -and -not $processCapture.TimedOut -and $processCapture.ExitCode -eq 0 -and [int]$parsedOuter.target_exit_code -eq 0 -and $null -ne $parsedOuter.target_receipt
        $terminal = [ordered]@{
            implementation_commit = $ImplementationCommit
            mode = $Mode
            outer = [ordered]@{ bytes = $outerRaw.Length; sha256 = (Get-Sha256Hex -Bytes $outerRaw) }
            preregistration = [ordered]@{ blob = $PreregBlob; commit = $PreregCommit }
            process_exit_code = [int]$processCapture.ExitCode
            raw_stderr_retained = $false
            recorded_on = $RecordedOn
            schema_version = "small-ranker-v2.21-durable-terminal.v1"
            status = $(if ($complete) { "COMPLETE" } else { "INVALID_ONE_SHOT_CONSUMED" })
            target_exit_code = $(if ($null -ne $parsedOuter) { [int]$parsedOuter.target_exit_code } else { 2 })
            target_receipt = $(if ($null -ne $parsedOuter) { $parsedOuter.target_receipt } else { $null })
        }
        if (-not $complete) {
            $terminal["error_code"] = $(if ($outerIsRawEnvelope) { "TARGET_OR_BOOTSTRAP_FAILED" } else { "OUTER_CAPTURE_FAILED" })
        }
        Assert-ReceiptPrivacy -Value $terminal
        $finalBytes = Get-CanonicalBytes -Value $terminal
        if ($finalBytes.Length -gt $MaximumTerminalBytes) { Throw-Code "TERMINAL_SIZE_GATE" }
        Write-ExclusiveBytes -Path ([string]$attemptPaths["result"]) -Bytes $finalBytes
        $terminalWritten = $true
        $finalExitCode = $(if ($complete) { 0 } else { 2 })
    }
    catch {
        $code = Get-SafeErrorCode -Caught $_
        if (-not $outerWritten) {
            try {
                $exitCode = if ($null -ne $processCapture) { [int]$processCapture.ExitCode } else { 2 }
                $stdoutBytes = if ($null -ne $processCapture) { [int]$processCapture.Stdout.Length } else { 0 }
                $stderrBytes = if ($null -ne $processCapture) { [int]$processCapture.Stderr.Length } else { 0 }
                $outerRaw = Get-CanonicalBytes -Value (New-SanitizedOuterFailure -Code $code -ProcessExitCode $exitCode -StdoutBytes $stdoutBytes -StderrBytes $stderrBytes)
                Write-ExclusiveBytes -Path ([string]$attemptPaths["outer"]) -Bytes $outerRaw
                $outerWritten = $true
            }
            catch {
                # The fixed path may have appeared after claim. Never read,
                # truncate, replace, or delete it; continue to the terminal.
            }
        }
        if (-not $terminalWritten) {
            $invalid = [ordered]@{
                error_code = $code
                implementation_commit = $ImplementationCommit
                mode = $Mode
                outer_capture_durable = $outerWritten
                preregistration = [ordered]@{ blob = $PreregBlob; commit = $PreregCommit }
                raw_stderr_retained = $false
                recorded_on = $RecordedOn
                schema_version = "small-ranker-v2.21-durable-terminal.v1"
                status = "INVALID_ONE_SHOT_CONSUMED"
            }
            if ($outerWritten -and $null -ne $outerRaw) {
                $invalid["outer"] = [ordered]@{ bytes = $outerRaw.Length; sha256 = (Get-Sha256Hex -Bytes $outerRaw) }
            }
            try {
                Assert-ReceiptPrivacy -Value $invalid
                $finalBytes = Get-CanonicalBytes -Value $invalid
                Write-ExclusiveBytes -Path ([string]$attemptPaths["result"]) -Bytes $finalBytes
                $terminalWritten = $true
            }
            catch {
                $finalBytes = Get-CanonicalBytes -Value ([ordered]@{
                    error_code = "TERMINAL_DURABLE_WRITE_FAILED"
                    implementation_commit = $ImplementationCommit
                    mode = $Mode
                    recorded_on = $RecordedOn
                    schema_version = "small-ranker-v2.21-ephemeral-failure.v1"
                    status = "ONE_SHOT_CONSUMED_CRASH"
                })
            }
        }
        $finalExitCode = 2
    }
}
catch {
    $code = Get-SafeErrorCode -Caught $_
    if ($claimed) {
        # This branch is defensive: every post-claim operation is handled by
        # the inner transaction above.
        $status = "ONE_SHOT_CONSUMED_CRASH"
    }
    else {
        $status = "PRELAUNCH_BLOCKED_NOT_CONSUMED"
    }
    $finalBytes = Get-CanonicalBytes -Value ([ordered]@{
        error_code = $code
        implementation_commit = $ImplementationCommit
        mode = $Mode
        recorded_on = $RecordedOn
        schema_version = "small-ranker-v2.21-ephemeral-failure.v1"
        status = $status
    })
    $finalExitCode = 2
}

if ($null -eq $finalBytes) {
    $finalBytes = Get-CanonicalBytes -Value ([ordered]@{
        error_code = "MISSING_TERMINAL_BYTES"
        implementation_commit = $ImplementationCommit
        mode = $Mode
        recorded_on = $RecordedOn
        schema_version = "small-ranker-v2.21-ephemeral-failure.v1"
        status = $(if ($claimed) { "ONE_SHOT_CONSUMED_CRASH" } else { "PRELAUNCH_BLOCKED_NOT_CONSUMED" })
    })
    $finalExitCode = 2
}

Write-ConsoleCanonical -Bytes $finalBytes
exit $finalExitCode
