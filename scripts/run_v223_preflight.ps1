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

# This is the sole external formal orchestrator for SR-v2.23.  It intentionally
# has no data-set paths: all data/source gates remain inside the source-only
# Python runner.  The outer process owns the irreversible claim -> raw outer ->
# terminal transaction so a non-zero nested exit can never discard its receipt.

$ProjectRoot = "D:\tiktok\techjam-v2-23-oov-chargram-bridge"
$ProjectRootPosix = "D:/tiktok/techjam-v2-23-oov-chargram-bridge"
$RuntimeBase = "D:\tiktok\.v223_runtime"
$RuntimeBasePosix = "D:/tiktok/.v223_runtime"
$PythonExe = "D:\450\conda\envs\tiktok\python.exe"
$PythonBytes = 93184L
$PythonSha256 = "7819c841b9a6457da034e567563de1283dbc0b86482fd83d62b5d982d2a83a63"
$PowerShellExe = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$PowerShellBytes = 454656L
$PowerShellSha256 = "7600ffe12da441fe89d035b13801e8e91d064bc544a27b19a5cf49f6ab8b18f5"
$PowerShellVersion = "5.1.26100.9168"
$GitExe = "C:\Program Files\Git\mingw64\bin\git.exe"
$GitBytes = 4018680L
$GitSha256 = "3fe4878d8399f6fb7632b9325559d1bb38c3a17aac7a60f667c1e5f90b865248"
$GitVersion = "git version 2.45.2.windows.1"
$GitDir = "D:\tiktok\techjam-err402\.git\worktrees\techjam-v2-23-oov-chargram-bridge"
$GitDirPosix = "D:/tiktok/techjam-err402/.git/worktrees/techjam-v2-23-oov-chargram-bridge"
$Branch = "small-ranker-v2.23-oov-chargram-bridge"
$ParentCommit = "e6907aa322d5ae9b3a51341cc01c5b67fbd1b2fd"
$PreregCommit = "2f8a9dbed16b8e6c4de701f15cf4e63a4d43f78c"
$PreregBlob = "0f875b691f6a373433895daa5d62b594e0ceca2e"
$PreregBytes = 43726L
$PreregSha256 = "dce8ecdf7474ae6494cb46bc226e09e4876a5020f183c6c4a429346908e0d05f"
$ParentOracleBlobOid = "fcc1c98aadbd0b9f94fd3dcfdd5f9bff61cb2d25"
$AttributesOracleBlobOid = "92260323f077c9861aa4edd5242aff772c875760"
$P8NegativeOracleBlobOid = "719078234dba297ce59f68d8a2b1734ec53c9c63"
$SlotLedgerOracleBlobOid = "72975cff12af59e4044e52911c58294cd74a785a"
$OracleFixtureCount = 101L
$OracleComparisonCount = 570L
$OracleMatrixSha256 = "f76d5c1d6678a9ac0bf56188ca013fb54a809da3e5d9b59c74b3306e135c312e"
$ProbeSchemaVersion = "small-ranker-v2.23-oov-chargram-lexicon-bridge-g0-probe.v1"
$RemoteUrl = "https://github.com/lamperriat/techjam-err402.git"
$PreregRelative = "configs/small_ranker_v2_23.oov_chargram_lexicon_bridge_g0_preregistration.json"
$BootstrapRelative = "scripts/v223_safe_bootstrap.py"
$RunnerRelative = "scripts/probe_oov_chargram_bridge_g0.py"
$WorkerRelative = "scripts/oov_chargram_bridge_g0_worker.py"
$UnionRelative = "starter/oov_chargram_bridge_g0.py"
$PowerShellRelative = "scripts/run_v223_preflight.ps1"
$TestRelative = "tests/test_oov_chargram_bridge_g0.py"
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
$WorkerFailureSchemaVersion = "small-ranker-v2.23-oov-chargram-worker-failure.v1"
$RunnerFailureSchemaVersion = "small-ranker-v2.23-oov-chargram-runner-failure.v1"
$InvalidTerminalSchemaVersion = "small-ranker-v2.23-oov-chargram-terminal-invalid.v1"
$ZeroSha256 = "0000000000000000000000000000000000000000000000000000000000000000"
$StageIds = @(
    "preflight_20_uncached_direct", "preflight_20_cached_direct", "preflight_20_cached_module",
    "preflight_100_uncached_direct", "preflight_100_cached_direct", "preflight_100_cached_module",
    "candidate_2000_cached_direct", "candidate_2000_cached_module", "preclaim_synthetic"
)
$WorkerPhases = @(
    "ARGUMENT_VALIDATION", "ENVIRONMENT_AUDIT", "SOURCE_VALIDATION", "TRACE_PREPARATION",
    "SEALED_SOURCE_VALIDATION", "LEXICON_INITIALIZATION", "TRAJECTORY", "SQLITE_CLOSE",
    "SOURCE_REVALIDATION", "RESOURCE_VALIDATION", "TRACE_PUBLICATION", "UNKNOWN"
)
$WorkerErrorCodes = @(
    "ARGUMENT_INVALID", "ENVIRONMENT_INVALID", "SOURCE_IDENTITY", "SOURCE_IMPORT", "TRACE_PATH",
    "SEALED_SOURCE_SCHEMA", "SEALED_SOURCE_IDENTITY", "LEXICON_SCHEMA", "LEXICON_RESOURCE",
    "QUERY_ONLY_CONTRACT", "CONTEXT_SCHEMA", "C200_SCHEMA", "EXPANSION_CONTRACT", "CACHE_CONTRACT",
    "RESOURCE_GATE", "NETWORK_ATTEMPT", "GPU_RUNTIME_PRESENT", "SQLITE_CLOSE", "TRACE_PUBLICATION",
    "PRIVACY_SCAN", "INTERNAL_INVARIANT", "UNCLASSIFIED", "UNAVAILABLE"
)
$RunnerErrorCodes = @(
    "WORKER_FAILURE", "WORKER_EXIT_UNVALIDATED", "WORKER_STDERR", "WORKER_TIMEOUT",
    "WORKER_RECEIPT_SCHEMA", "WORKER_TRACE", "REPEAT_MISMATCH", "RESOURCE_GATE",
    "SOURCE_IDENTITY", "GIT_IDENTITY", "PREFLIGHT_NO_INFORMATION", "TARGET_ATTACH",
    "AGGREGATE_SCHEMA", "PRIVACY_SCAN", "INTERNAL_INVARIANT", "UNCLASSIFIED"
)
$PowerShellErrorCodes = @(
    "BOOTSTRAP_FAILURE", "RUNNER_FAILURE", "OUTER_CAPTURE_FAILURE", "OUTER_SCHEMA_FAILURE",
    "TRANSACTION_FAILURE", "UNCLASSIFIED"
)
$CommonGitDir = "D:\tiktok\techjam-err402\.git"
$CriticalBinaries = @(
    [pscustomobject]@{ Path = "D:\450\conda\envs\tiktok\python311.dll"; Bytes = 6193152L; Sha256 = "08701864dea4e08c077c1c5bc6cb208d5628dba9473a318fa6ce3796a86806c5" },
    [pscustomobject]@{ Path = "D:\450\conda\envs\tiktok\DLLs\_sqlite3.pyd"; Bytes = 99328L; Sha256 = "67858a7dcbce3abef73328276c19dc06abe2515ab01d7cbc7fc7ad7bd3e2114c" },
    [pscustomobject]@{ Path = "D:\450\conda\envs\tiktok\Library\bin\sqlite3.dll"; Bytes = 3239936L; Sha256 = "9519340d2ede13b05cd889605e3a46cc6bde702f266061266e22c18a0951a04e" }
)
$GitControlFiles = @(
    [pscustomobject]@{ Path = (Join-Path $GitDir "gitdir"); Bytes = 49L; Sha256 = "a66dda9b3a46f0841f772867c30df2bbac07b8bde0fecedfab129c0756cb421b" },
    [pscustomobject]@{ Path = (Join-Path $GitDir "commondir"); Bytes = 6L; Sha256 = "340ddcb67a6204f742cd1e28e5b462622dde7daaa8ee36001897196aacdc6d47" },
    [pscustomobject]@{ Path = (Join-Path $GitDir "HEAD"); Bytes = 55L; Sha256 = "986e4bd7ff8eb41c11fe49e77c277e605cbaf2c7b34f3d27b546f8dd90d777c1" }
)
$ExpectedCommonConfigProjection = @(
    "core.bare=false", "core.filemode=false", "core.ignorecase=true",
    "core.logallrefupdates=true", "core.repositoryformatversion=0", "core.symlinks=false",
    "remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*",
    "remote.origin.url=https://github.com/lamperriat/techjam-err402.git",
    "remote.upstream.fetch=+refs/heads/*:refs/remotes/upstream/*",
    "remote.upstream.url=https://github.com/TechJam2026/techjam-conversational-search.git"
)

$FastTrack = Join-Path $ProjectRoot "experiments\fast_track"
$PreregPath = Join-Path $ProjectRoot ($PreregRelative -replace "/", "\")
$BootstrapPath = Join-Path $ProjectRoot ($BootstrapRelative -replace "/", "\")
$RunnerPath = Join-Path $ProjectRoot ($RunnerRelative -replace "/", "\")
$WorkerPath = Join-Path $ProjectRoot ($WorkerRelative -replace "/", "\")
$WorktreeDotGit = Join-Path $ProjectRoot ".git"

$ModePaths = @{
    preflight = [ordered]@{
        claim = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_preflight_claim_20260831.json"
        outer = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_preflight_outer_20260831.json"
        result = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_preflight_20260831.json"
    }
    candidate = [ordered]@{
        claim = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_candidate_recall_claim_20260831.json"
        outer = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_candidate_recall_outer_20260831.json"
        result = Join-Path $FastTrack "small_ranker_v2_23_oov_chargram_bridge_g0_candidate_recall_20260831.json"
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
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($algorithm.ComputeHash($Bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $algorithm.Dispose()
    }
}

function Get-GitBlobHex {
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes)
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
    param([Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Bytes)
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
        if ([string]::IsNullOrEmpty($value)) { Throw-Code "WINDOWS_BOOTSTRAP_ENVIRONMENT" }
        $environment[$name] = $value
    }
    $systemRoot = [System.Environment]::GetEnvironmentVariable("SYSTEMROOT")
    if ([string]::IsNullOrEmpty($systemRoot)) { Throw-Code "WINDOWS_BOOTSTRAP_ENVIRONMENT" }
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
    $environment["GIT_CONFIG_COUNT"] = "0"
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
    Assert-GitOperationAllowed -Arguments $Arguments
    $before = Get-GitControlCheckpoint
    $prefix = @(
        "--no-pager",
        "--no-replace-objects",
        "--no-optional-locks",
        "--git-dir=$GitDirPosix",
        "--work-tree=$ProjectRootPosix",
        "-c", "core.hooksPath=NUL",
        "-c", "core.attributesFile=NUL",
        "-c", "include.path=/dev/null"
    )
    $captured = Invoke-CapturedProcess -FileName $GitExe -Arguments ($prefix + $Arguments) -WorkingDirectory "C:\Windows\System32" -Environment (New-GitEnvironment) -TimeoutMilliseconds 30000
    $after = Get-GitControlCheckpoint
    if ($before -cne $after -or $captured.TimedOut -or $captured.Stdout.Length -gt 8388608 -or $captured.Stderr.Length -ne 0 -or $AllowedExitCodes -notcontains $captured.ExitCode) {
        Throw-Code "GIT_COMMAND_FAILED"
    }
    $text = Convert-Utf8Strict -Bytes $captured.Stdout
    return [pscustomobject]@{ Bytes = $captured.Stdout; ExitCode = $captured.ExitCode; Text = $text }
}

function Assert-GitOperationAllowed {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $shape = $Arguments -join " "
    $allowed = (
        $shape -ceq "--version" -or
        $shape -cmatch "^rev-parse --verify (?:[0-9a-f]{40}|refs/heads/$([regex]::Escape($Branch))|refs/remotes/origin/$([regex]::Escape($Branch)))\^\{commit\}$" -or
        $shape -cmatch "^rev-parse [0-9a-f]{40}\^$" -or
        $shape -cmatch "^merge-base --is-ancestor [0-9a-f]{40} [0-9a-f]{40}$" -or
        $shape -cmatch "^rev-list --min-parents=2 [0-9a-f]{40}\.\.[0-9a-f]{40}$" -or
        $shape -cmatch "^diff-tree --no-commit-id --name-only --no-renames -r [0-9a-f]{40} [0-9a-f]{40}$" -or
        $shape -cmatch "^cat-file -t [0-9a-f]{40}$" -or
        $shape -cmatch "^cat-file blob [0-9a-f]{40}:[A-Za-z0-9_./-]+$"
    )
    if (-not $allowed -or $Arguments[0] -in @("status", "diff", "ls-files", "fetch", "pull", "reset", "rebase", "commit", "push")) {
        Throw-Code "GIT_COMMAND_DENIED"
    }
}

function Get-CommonConfigProjection {
    param([Parameter(Mandatory = $true)][byte[]]$Raw)
    $text = (Convert-Utf8Strict -Bytes $Raw).Replace("`r`n", "`n")
    $section = ""
    $subsection = ""
    $projected = @{}
    foreach ($rawLine in $text.Split("`n")) {
        $line = $rawLine.Trim()
        if ([string]::IsNullOrEmpty($line) -or $line.StartsWith("#") -or $line.StartsWith(";")) { continue }
        $header = [regex]::Match($line, '^\[([A-Za-z0-9.-]+)(?:\s+"([A-Za-z0-9._/-]+)")?\]$')
        if ($header.Success) {
            $section = $header.Groups[1].Value.ToLowerInvariant()
            $subsection = $header.Groups[2].Value.ToLowerInvariant()
            if (@("core", "remote", "branch") -cnotcontains $section) { Throw-Code "GIT_CONFIG_SECTION_DENIED" }
            if ($section -eq "core" -and -not [string]::IsNullOrEmpty($subsection)) { Throw-Code "GIT_CONFIG_SECTION_DENIED" }
            if ($section -eq "remote" -and @("origin", "upstream") -cnotcontains $subsection) { Throw-Code "GIT_CONFIG_REMOTE_DENIED" }
            if ($section -eq "branch" -and [string]::IsNullOrEmpty($subsection)) { Throw-Code "GIT_CONFIG_BRANCH_SCHEMA" }
            continue
        }
        $setting = [regex]::Match($line, '^([A-Za-z0-9.-]+)\s*=\s*(.*)$')
        if (-not $setting.Success -or [string]::IsNullOrEmpty($section)) { Throw-Code "GIT_CONFIG_SYNTAX" }
        if ($section -eq "branch") { continue }
        $key = if ($section -eq "core") {
            "core." + $setting.Groups[1].Value.ToLowerInvariant()
        }
        else {
            "remote." + $subsection + "." + $setting.Groups[1].Value.ToLowerInvariant()
        }
        if ($projected.ContainsKey($key)) { Throw-Code "GIT_CONFIG_DUPLICATE" }
        $projected[$key] = $setting.Groups[2].Value.Trim()
    }
    $lines = @($projected.Keys | ForEach-Object { $_ + "=" + [string]$projected[$_] } | Sort-Object)
    if ($lines.Count -ne $ExpectedCommonConfigProjection.Count) { Throw-Code "GIT_CONFIG_PROJECTION" }
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ([string]$lines[$index] -cne [string]$ExpectedCommonConfigProjection[$index]) { Throw-Code "GIT_CONFIG_PROJECTION" }
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n") + "`n")
    if ($bytes.Length -ne 403 -or (Get-Sha256Hex -Bytes $bytes) -cne "b01fd6e872247b479dd0523a4901c710114a94bdf05333cd6afb34cebae4aaf0") {
        Throw-Code "GIT_CONFIG_PROJECTION"
    }
    return "projection|403|b01fd6e872247b479dd0523a4901c710114a94bdf05333cd6afb34cebae4aaf0"
}

function Assert-FormalPowerShellHost {
    param(
        [Parameter(Mandatory = $true)][string]$RequestedMode,
        [Parameter(Mandatory = $true)][string]$RequestedCommit
    )
    if (
        $RequestedMode -cnotmatch "^(preflight|candidate)$" -or
        $RequestedCommit -cnotmatch "^[0-9a-f]{40}$"
    ) { Throw-Code "POWERSHELL_ARGUMENT_CASE" }
    $processPath = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    if ((Get-FullPathKey $processPath) -cne (Get-FullPathKey $PowerShellExe)) {
        Throw-Code "POWERSHELL_EXECUTABLE_DRIFT"
    }
    $raw = Get-PlainFileBytes -Path $PowerShellExe -MaximumBytes 1048576
    if (
        $raw.LongLength -ne $PowerShellBytes -or
        (Get-Sha256Hex -Bytes $raw) -cne $PowerShellSha256 -or
        $PSVersionTable.PSVersion.ToString() -cne $PowerShellVersion
    ) {
        Throw-Code "POWERSHELL_IDENTITY_DRIFT"
    }
    $argv = @([Environment]::GetCommandLineArgs())
    if (
        $argv.Count -ne 12 -or
        (Get-FullPathKey ([string]$argv[0])) -cne (Get-FullPathKey $PowerShellExe) -or
        [string]$argv[1] -cne "-NoLogo" -or
        [string]$argv[2] -cne "-NoProfile" -or
        [string]$argv[3] -cne "-NonInteractive" -or
        [string]$argv[4] -cne "-ExecutionPolicy" -or
        [string]$argv[5] -cne "Bypass" -or
        [string]$argv[6] -cne "-File" -or
        (Get-FullPathKey ([string]$argv[7])) -cne (Get-FullPathKey $PSCommandPath) -or
        [string]$argv[8] -cne "-Mode" -or
        [string]$argv[9] -cne $RequestedMode -or
        [string]$argv[10] -cne "-ImplementationCommit" -or
        [string]$argv[11] -cne $RequestedCommit
    ) {
        Throw-Code "POWERSHELL_ARGV_DRIFT"
    }
}

function Get-GitControlCheckpoint {
    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $GitControlFiles) {
        $raw = Get-PlainFileBytes -Path ([string]$entry.Path) -MaximumBytes 8388608
        $sha = Get-Sha256Hex -Bytes $raw
        if ($raw.LongLength -ne [int64]$entry.Bytes -or $sha -cne [string]$entry.Sha256) {
            Throw-Code "GIT_CONTROL_PLANE_DRIFT"
        }
        $parts.Add(([string]$entry.Path) + "|" + $raw.LongLength + "|" + $sha)
    }
    $configRaw = Get-PlainFileBytes -Path (Join-Path $CommonGitDir "config") -MaximumBytes 8388608
    $projection = Get-CommonConfigProjection -Raw $configRaw
    $parts.Add((Join-Path $CommonGitDir "config") + "|" + $configRaw.LongLength + "|" + (Get-Sha256Hex -Bytes $configRaw))
    $parts.Add($projection)
    return ($parts -join "`n")
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
        "D:\tiktok\techjam-err402\.git\objects\info\alternates",
        "D:\tiktok\techjam-err402\.git\objects\info\http-alternates",
        "D:\tiktok\techjam-err402\.git\info\grafts",
        "D:\tiktok\techjam-err402\.git\refs\replace",
        "D:\tiktok\techjam-err402\.git\config.worktree"
    )) {
        if (Test-LiteralPathExists $forbidden) {
            Throw-Code "GIT_CONTROL_PLANE_UNSUPPORTED"
        }
    }
    Assert-ExecutableFingerprint -Path $GitExe -ExpectedBytes $GitBytes -ExpectedSha256 $GitSha256
    foreach ($binary in $CriticalBinaries) {
        Assert-ExecutableFingerprint -Path ([string]$binary.Path) -ExpectedBytes ([int64]$binary.Bytes) -ExpectedSha256 ([string]$binary.Sha256)
    }
    [void](Get-GitControlCheckpoint)
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
    $object = Invoke-FrozenGit -Arguments @("cat-file", "blob", "$Commit`:$RelativePath")
    $value = Get-GitBlobHex -Bytes $object.Bytes
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
    $correctionChain = @(
        [pscustomobject]@{ Child = $PreregCommit; Parent = $ParentCommit }
    )
    foreach ($edge in $correctionChain) {
        $resolvedCorrection = Get-GitSingleLine -Arguments @("rev-parse", "--verify", "$($edge.Child)^{commit}")
        $resolvedParent = Get-GitSingleLine -Arguments @("rev-parse", "$($edge.Child)^")
        if ($resolvedCorrection -cne [string]$edge.Child -or $resolvedParent -cne [string]$edge.Parent) {
            Throw-Code "PREREG_CORRECTION_CHAIN_DRIFT"
        }
    }
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
    $preregRaw = Convert-CrlfToLf -Bytes (Get-PlainFileBytes -Path $PreregPath -MaximumBytes 1048576)
    if ($preregRaw.LongLength -ne $PreregBytes -or (Get-Sha256Hex -Bytes $preregRaw) -cne $PreregSha256) {
        Throw-Code "PREREG_RAW_IDENTITY_DRIFT"
    }

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
        [Parameter(Mandatory = $true)][byte[]]$Bytes,
        [ref]$CreationObserved
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
    # A successful CreateNew is the irreversible boundary.  The caller that
    # owns a one-shot claim receives this signal before any write, flush,
    # close, reread, or hash operation can fail.
    if ($null -ne $CreationObserved) { $CreationObserved.Value = $true }
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
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Stdout,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][byte[]]$Stderr,
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
        [Parameter(Mandatory = $true)][ValidateSet("direct", "module")][string]$ExpectedMode,
        [Parameter(Mandatory = $true)][string]$ExpectedPycache
    )
    $text = Convert-Utf8Strict -Bytes $Raw
    try { $value = $text.TrimEnd("`n") | ConvertFrom-Json -ErrorAction Stop } catch { Throw-Code "OUTER_JSON_PARSE" }
    Assert-ExactNames -Value $value -Expected @("bootstrap", "target_exit_code", "target_receipt")
    Assert-ExactNames -Value $value.bootstrap -Expected @("bootstrap_blob", "guarded_path", "mode", "pycache_prefix", "source_only", "target_blob")
    if (
        [string]$value.bootstrap.bootstrap_blob -cne $ExpectedBootstrapBlob -or
        [string]$value.bootstrap.target_blob -cne $ExpectedRunnerBlob -or
        [string]$value.bootstrap.mode -cne $ExpectedMode -or
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

function Assert-RunnerFailureReceipt {
    param([Parameter(Mandatory = $true)]$Receipt)
    Assert-ExactNames -Value $Receipt -Expected @(
        "canonical_failure_receipt", "child_exit_code", "failure_origin",
        "failure_site_id", "kind", "progress_bucket", "rss_bucket",
        "runner_error_code", "schema_version", "stack_hash", "stage_id",
        "status", "stderr_nonempty", "wall_time_bucket", "worker_error_code",
        "worker_phase"
    )
    if (
        [string]$Receipt.schema_version -cne $RunnerFailureSchemaVersion -or
        [string]$Receipt.status -cne "ERROR" -or [string]$Receipt.kind -cne "failure" -or
        @("worker", "runner") -cnotcontains [string]$Receipt.failure_origin -or
        $Receipt.canonical_failure_receipt -isnot [bool] -or
        $Receipt.stderr_nonempty -isnot [bool] -or
        -not ($Receipt.child_exit_code -is [int] -or $Receipt.child_exit_code -is [long]) -or
        [int64]$Receipt.child_exit_code -lt -1 -or [int64]$Receipt.child_exit_code -gt 255 -or
        $StageIds -cnotcontains [string]$Receipt.stage_id -or
        $WorkerPhases -cnotcontains [string]$Receipt.worker_phase -or
        $WorkerErrorCodes -cnotcontains [string]$Receipt.worker_error_code -or
        $RunnerErrorCodes -cnotcontains [string]$Receipt.runner_error_code -or
        @("NONE", "PARTIAL", "COMPLETE", "UNKNOWN") -cnotcontains [string]$Receipt.progress_bucket -or
        @("LT_1S", "1_TO_10S", "10_TO_60S", "60_TO_300S", "GE_300S", "UNKNOWN") -cnotcontains [string]$Receipt.wall_time_bucket -or
        @("LE_256M", "LE_512M", "LE_1G", "LE_1_5G", "GT_1_5G", "UNKNOWN") -cnotcontains [string]$Receipt.rss_bucket -or
        [string]$Receipt.failure_site_id -cnotmatch "^SITE_(0000|00([0-9][1-9]|[1-9][0-9]))$" -or
        [string]$Receipt.stack_hash -cnotmatch "^[0-9a-f]{64}$"
    ) { Throw-Code "RUNNER_FAILURE_SCHEMA" }
    if ([string]$Receipt.failure_site_id -ceq "SITE_0000" -and [string]$Receipt.stack_hash -cne $ZeroSha256) {
        Throw-Code "RUNNER_FAILURE_SCHEMA"
    }
    if ($Receipt.canonical_failure_receipt) {
        if (
            [string]$Receipt.failure_origin -cne "worker" -or
            [int64]$Receipt.child_exit_code -ne 1 -or
            [string]$Receipt.runner_error_code -cne "WORKER_FAILURE" -or
            $Receipt.stderr_nonempty -ne $false -or
            [string]$Receipt.worker_error_code -ceq "UNAVAILABLE"
        ) { Throw-Code "RUNNER_FAILURE_PROPAGATION" }
    }
    else {
        if (
            [string]$Receipt.failure_origin -cne "runner" -or
            [string]$Receipt.failure_site_id -cne "SITE_0000" -or
            [string]$Receipt.stack_hash -cne $ZeroSha256 -or
            [string]$Receipt.worker_error_code -cne "UNAVAILABLE" -or
            [string]$Receipt.worker_phase -cne "UNKNOWN" -or
            [string]$Receipt.progress_bucket -cne "UNKNOWN" -or
            [string]$Receipt.wall_time_bucket -cne "UNKNOWN" -or
            [string]$Receipt.rss_bucket -cne "UNKNOWN"
        ) { Throw-Code "RUNNER_FAILURE_SANITIZATION" }
    }
    Assert-ReceiptPrivacy -Value $Receipt
    return $Receipt
}

function Get-StdoutBytesBucket {
    param([int64]$Bytes)
    if ($Bytes -le 0) { return "ZERO" }
    if ($Bytes -le 1024) { return "LE_1K" }
    if ($Bytes -le 16384) { return "LE_16K" }
    return "GT_16K"
}

function Get-BoundedProcessExitCode {
    param([int64]$Value)
    if ($Value -ge 0 -and $Value -le 255) { return [int]$Value }
    return -1
}

function New-InvalidTerminal {
    param(
        [Parameter(Mandatory = $true)][string]$PowerShellCode,
        [string]$RootFailureOrigin = "unknown",
        [string]$RootStageId = "UNKNOWN",
        [string]$RootErrorCode = "UNCLASSIFIED",
        [bool]$CanonicalFailureReceipt = $false
    )
    if ($PowerShellErrorCodes -cnotcontains $PowerShellCode) { $PowerShellCode = "UNCLASSIFIED" }
    if (@("worker", "runner", "bootstrap", "powershell", "unknown") -cnotcontains $RootFailureOrigin) {
        $RootFailureOrigin = "unknown"
    }
    if ($RootStageId -cne "UNKNOWN" -and $StageIds -cnotcontains $RootStageId) { $RootStageId = "UNKNOWN" }
    if ($RootErrorCode -cnotmatch "^[A-Z][A-Z0-9_]{1,79}$") { $RootErrorCode = "UNCLASSIFIED" }
    $claimSha = if ($null -ne $claimIdentity -and [string]$claimIdentity.sha256 -cmatch "^[0-9a-f]{64}$") { [string]$claimIdentity.sha256 } else { $ZeroSha256 }
    $outerSha = if ($null -ne $outerRaw) { Get-Sha256Hex -Bytes $outerRaw } else { $ZeroSha256 }
    $processExit = if ($null -ne $processCapture) { Get-BoundedProcessExitCode -Value ([int64]$processCapture.ExitCode) } else { -1 }
    $stderrPresent = $null -ne $processCapture -and $processCapture.Stderr.Length -ne 0
    $stdoutLength = if ($null -ne $processCapture) { [int64]$processCapture.Stdout.Length } else { 0L }
    $value = [ordered]@{
        candidate_commit = $ImplementationCommit
        canonical_failure_receipt = $CanonicalFailureReceipt
        claim_sha256 = $claimSha
        error_code = $PowerShellCode
        mode = $Mode
        outer_sha256 = $outerSha
        process_exit_code = $processExit
        root_error_code = $RootErrorCode
        root_failure_origin = $RootFailureOrigin
        root_stage_id = $RootStageId
        schema_version = $InvalidTerminalSchemaVersion
        status = "INVALID_ONE_SHOT_CONSUMED"
        stderr_nonempty = [bool]$stderrPresent
        stdout_bytes_bucket = Get-StdoutBytesBucket -Bytes $stdoutLength
    }
    Assert-ExactNames -Value ([pscustomobject]$value) -Expected @(
        "candidate_commit", "canonical_failure_receipt", "claim_sha256", "error_code",
        "mode", "outer_sha256", "process_exit_code", "root_error_code",
        "root_failure_origin", "root_stage_id", "schema_version", "status",
        "stderr_nonempty", "stdout_bytes_bucket"
    )
    Assert-ReceiptPrivacy -Value $value
    return $value
}

function New-SanitizedOuterFailure {
    param(
        [Parameter(Mandatory = $true)][string]$Code,
        [int]$ProcessExitCode = 2,
        [int]$StdoutBytes = 0,
        [int]$StderrBytes = 0
    )
    $boundedExit = Get-BoundedProcessExitCode -Value ([int64]$ProcessExitCode)
    return [ordered]@{
        capture_status = "NO_VALID_CANONICAL_ENVELOPE"
        error_code = $Code
        process_exit_code = $boundedExit
        raw_stderr_retained = $false
        raw_stdout_retained = $false
        schema_version = "small-ranker-v2.23-outer-capture-failure.v1"
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
        [System.IO.Path]::GetFileName($runtimeRoot) -cnotmatch "^v223-[0-9a-f]{32}$" -or
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
        [string]$claim.schema_version -cne "small-ranker-v2.23-durable-one-shot-claim.v1" -or
        [string]$claim.experiment_id -cne "SR-V2.23-TARGET-BLIND-OOV-CHARGRAM-LEXICON-BRIDGE-G0" -or
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
        "resources", "rerun_forbidden", "runtime", "schema_version", "sources", "stages", "status"
    )
    Assert-IdentityEqual -Observed $targetReceipt.claim -Expected $claimIdentity -Code "PREFLIGHT_RECEIPT_CLAIM_BINDING"
    Assert-CanonicalObjectEqual -Left $targetReceipt.bootstrap -Right $outer.bootstrap -Code "PREFLIGHT_RECEIPT_BOOTSTRAP_BINDING"
    Assert-ExactNames -Value $targetReceipt.implementation -Expected @("branch", "commit", "default_off", "preregistration_commit", "served_top10_unchanged", "target_blind")
    Assert-ExactNames -Value $targetReceipt.integrity -Expected @("exact_triplet_each_stage", "legacy_route_executions", "network_attempt_count", "ordered_variable_c200_prefix", "target_sources_opened")
    Assert-ExactNames -Value $targetReceipt.device -Expected @("gpu_peak_bytes", "reason", "selected")
    Assert-ExactNames -Value $targetReceipt.entrypoint_regression -Expected @("legacy_module_denied_direct", "legacy_module_denied_module", "runner_direct", "runner_module", "worker_direct", "worker_module")
    Assert-ExactNames -Value $targetReceipt.sources -Expected @("catalog", "sealed_c200", "visible_context")
    Assert-ExactNames -Value $targetReceipt.resources -Expected @(
        "free_disk_bytes_before_formal",
        "limit100_cached_direct_plus_module_parent_wall_seconds",
        "limit100_linear_extrapolation_x1_5_seconds"
    )
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
        [string]$targetReceipt.experiment_id -cne "SR-V2.23-TARGET-BLIND-OOV-CHARGRAM-LEXICON-BRIDGE-G0" -or
        [string]$targetReceipt.mode -cne "preflight" -or [string]$targetReceipt.status -cne "TARGET_FREE_PREFLIGHT_COMPLETE" -or
        [string]$targetReceipt.recorded_on -cne $RecordedOn -or $targetReceipt.rerun_forbidden -isnot [bool] -or $targetReceipt.rerun_forbidden -ne $true -or
        [string]$targetReceipt.implementation.branch -cne $Branch -or [string]$targetReceipt.implementation.commit -cne $Commit -or
        [string]$targetReceipt.implementation.preregistration_commit -cne $PreregCommit -or
        $targetReceipt.implementation.default_off -ne $true -or $targetReceipt.implementation.target_blind -ne $true -or $targetReceipt.implementation.served_top10_unchanged -ne $true -or
        $targetReceipt.integrity.exact_triplet_each_stage -ne $true -or $targetReceipt.integrity.ordered_variable_c200_prefix -ne $true -or
        [int64]$targetReceipt.integrity.legacy_route_executions -ne 0 -or $targetReceipt.integrity.target_sources_opened -ne $false -or [int64]$targetReceipt.integrity.network_attempt_count -ne 0 -or
        [string]$targetReceipt.device.selected -cne "CPU" -or [string]$targetReceipt.device.reason -cne "frozen SQLite FTS5 chargram/edit-distance bridge backend" -or [int64]$targetReceipt.device.gpu_peak_bytes -ne 0 -or
        $targetReceipt.entrypoint_regression.runner_direct -ne $true -or $targetReceipt.entrypoint_regression.runner_module -ne $true -or
        $targetReceipt.entrypoint_regression.worker_direct -ne $true -or $targetReceipt.entrypoint_regression.worker_module -ne $true -or
        $targetReceipt.entrypoint_regression.legacy_module_denied_direct -ne $true -or $targetReceipt.entrypoint_regression.legacy_module_denied_module -ne $true -or
        [string]$targetReceipt.git.branch -cne $Branch -or [string]$targetReceipt.git.commit -cne $Commit -or
        [string]$targetReceipt.git.preregistration_commit -cne $PreregCommit -or [string]$targetReceipt.git.remote -cne $RemoteUrl -or
        $targetReceipt.git.pushed -ne $true -or $targetReceipt.git.object_only_git -ne $true -or $stages.Count -ne 2 -or
        -not ($targetReceipt.resources.free_disk_bytes_before_formal -is [int] -or $targetReceipt.resources.free_disk_bytes_before_formal -is [long]) -or
        [int64]$targetReceipt.resources.free_disk_bytes_before_formal -lt 536870912L -or
        [double]$targetReceipt.resources.limit100_cached_direct_plus_module_parent_wall_seconds -gt 60.0 -or
        [double]$targetReceipt.resources.limit100_linear_extrapolation_x1_5_seconds -gt 1800.0
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
    if ($stages[1].information_available -isnot [bool] -or $stages[1].information_available -ne $true -or @($stages[1].no_information_reasons).Count -ne 0) {
        Throw-Code "PREFLIGHT_NO_INFORMATION"
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
        [string]$terminal.schema_version -cne "small-ranker-v2.23-durable-terminal.v1" -or
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

function Get-DirectoryStableIdentity {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ($null -eq ("V223DirectoryIdentity" -as [type])) {
        $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class V223DirectoryIdentity {
    [StructLayout(LayoutKind.Sequential)]
    private struct BY_HANDLE_FILE_INFORMATION {
        public uint FileAttributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;
        public uint VolumeSerialNumber;
        public uint FileSizeHigh;
        public uint FileSizeLow;
        public uint NumberOfLinks;
        public uint FileIndexHigh;
        public uint FileIndexLow;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern SafeFileHandle CreateFileW(
        string name, uint access, uint share, IntPtr security,
        uint creation, uint flags, IntPtr template);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(
        SafeFileHandle handle, out BY_HANDLE_FILE_INFORMATION information);

    public static string Read(string path) {
        const uint FILE_READ_ATTRIBUTES = 0x80;
        const uint FILE_SHARE_ALL = 0x7;
        const uint OPEN_EXISTING = 3;
        const uint FILE_FLAG_BACKUP_SEMANTICS = 0x02000000;
        using (SafeFileHandle handle = CreateFileW(
            path, FILE_READ_ATTRIBUTES, FILE_SHARE_ALL, IntPtr.Zero,
            OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, IntPtr.Zero)) {
            if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
            BY_HANDLE_FILE_INFORMATION info;
            if (!GetFileInformationByHandle(handle, out info))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return info.VolumeSerialNumber.ToString("x8") + ":" +
                   info.FileIndexHigh.ToString("x8") + ":" +
                   info.FileIndexLow.ToString("x8");
        }
    }
}
'@
        Add-Type -TypeDefinition $source -Language CSharp -ErrorAction Stop
    }
    Assert-PlainExistingPath -Path $Path -Kind Directory | Out-Null
    try { $identity = [V223DirectoryIdentity]::Read([System.IO.Path]::GetFullPath($Path)) } catch { Throw-Code "RUNTIME_IDENTITY_BACKEND" }
    if ([string]$identity -cnotmatch "^[0-9a-f]{8}:[0-9a-f]{8}:[0-9a-f]{8}$") {
        Throw-Code "RUNTIME_IDENTITY_BACKEND"
    }
    return [string]$identity
}

function New-FreshRuntime {
    Assert-PlainExistingPath -Path "D:\tiktok" -Kind Directory | Out-Null
    if (-not [System.IO.Directory]::Exists($RuntimeBase)) {
        [void][System.IO.Directory]::CreateDirectory($RuntimeBase)
    }
    Assert-PlainExistingPath -Path $RuntimeBase -Kind Directory | Out-Null
    $nonce = [System.Guid]::NewGuid().ToString("N")
    $runtimeRoot = Join-Path $RuntimeBase ("v223-" + $nonce)
    if (Test-LiteralPathExists $runtimeRoot) { Throw-Code "RUNTIME_COLLISION" }
    [void][System.IO.Directory]::CreateDirectory($runtimeRoot)
    Assert-PlainExistingPath -Path $runtimeRoot -Kind Directory | Out-Null
    $stableIdentity = Get-DirectoryStableIdentity -Path $runtimeRoot
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
    return [pscustomobject]@{ Root = $runtimeRoot; StableIdentity = $stableIdentity; Pycache = $pycache; Temp = $temp }
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
    if ((Get-DirectoryStableIdentity -Path $root) -cne [string]$Runtime.StableIdentity) {
        Throw-Code "RUNTIME_CLEANUP_IDENTITY"
    }
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
        if ($moduleChildren.Count -ne 1 -or $moduleChildren[0].Name -cne "v223_safe_bootstrap.py" -or -not ($moduleChildren[0] -is [System.IO.FileInfo])) {
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
            $modulePath = Join-Path $moduleRoot "v223_safe_bootstrap.py"
            $bootstrapBytes = Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608
            Write-ExclusiveBytes -Path $modulePath -Bytes $bootstrapBytes
            $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
            $arguments += @("-m", "v223_safe_bootstrap")
        }
        $arguments += @(
            "--mode", $InvocationMode,
            "--target-path", $TargetPath.Replace("\", "/"),
            "--target-module", $TargetModule,
            "--target-blob", $TargetBlob,
            "--bootstrap-blob", $BootstrapBlob,
            "--",
            "--entrypoint-self-check", "--require-module", "starter.oov_chargram_bridge_g0"
        )
        if ($TargetModule -ceq "scripts.oov_chargram_bridge_g0_worker") {
            $arguments += @("--stage-id", "preclaim_synthetic")
        }
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_ENTRYPOINT_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob $TargetBlob -ExpectedBootstrapBlob $BootstrapBlob -ExpectedMode $InvocationMode -ExpectedPycache $checkRuntime.Pycache
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
            [string]$outer.target_receipt.required_module -cne "starter.oov_chargram_bridge_g0" -or
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
        [pscustomobject]@{ Path = $RunnerPath; Module = "scripts.probe_oov_chargram_bridge_g0"; Relative = $RunnerRelative },
        [pscustomobject]@{ Path = $WorkerPath; Module = "scripts.oov_chargram_bridge_g0_worker"; Relative = $WorkerRelative }
    )
    foreach ($target in $targets) {
        foreach ($invocationMode in @("direct", "module")) {
            Invoke-PreclaimEntrypointCheck -InvocationMode $invocationMode -TargetPath $target.Path -TargetModule $target.Module -TargetBlob ([string]$Blobs[$target.Relative]) -BootstrapBlob ([string]$Blobs[$BootstrapRelative])
        }
    }
}

function Invoke-PreclaimOracleDifferentialChecks {
    param([Parameter(Mandatory = $true)]$Blobs)
    $receipts = @()
    foreach ($invocationMode in @("direct", "module")) {
        $checkRuntime = $null
        try {
            $checkRuntime = New-FreshRuntime
            $pycachePosix = $checkRuntime.Pycache.Replace("\", "/")
            $environment = New-OfflineEnvironment -TempPath $checkRuntime.Temp
            $arguments = @("-P", "-S", "-s", "-B", "-X", "pycache_prefix=$pycachePosix")
            if ($invocationMode -eq "direct") {
                $arguments += $BootstrapPath.Replace("\", "/")
            }
            else {
                $moduleRoot = Join-Path $checkRuntime.Root "module"
                [void][System.IO.Directory]::CreateDirectory($moduleRoot)
                Assert-PlainExistingPath -Path $moduleRoot -Kind Directory | Out-Null
                $modulePath = Join-Path $moduleRoot "v223_safe_bootstrap.py"
                Write-ExclusiveBytes -Path $modulePath -Bytes (Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608)
                $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
                $arguments += @("-m", "v223_safe_bootstrap")
            }
            $arguments += @(
                "--mode", $invocationMode,
                "--target-path", $RunnerPath.Replace("\", "/"),
                "--target-module", "scripts.probe_oov_chargram_bridge_g0",
                "--target-blob", [string]$Blobs[$RunnerRelative],
                "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
                "--", "--oracle-differential-self-check"
            )
            $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
            if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
                Throw-Code "PRECLAIM_ORACLE_ENVELOPE"
            }
            $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedMode $invocationMode -ExpectedPycache $checkRuntime.Pycache
            if ($capture.ExitCode -ne 0 -or [int64]$outer.target_exit_code -ne 0 -or $null -eq $outer.target_receipt) {
                Throw-Code "PRECLAIM_ORACLE_EXIT"
            }
            Assert-ExactNames -Value $outer.target_receipt -Expected @(
                "algorithm_fts_cache_privacy_pass", "comparison_count", "dependency_blob_oids",
                "exact_repeat", "fixture_count", "isolated_namespace", "legacy_runtime_absent",
                "matrix_sha256", "parent_blob_oid", "status"
            )
            Assert-ExactNames -Value $outer.target_receipt.dependency_blob_oids -Expected @(
                "starter/attributes.py", "starter/p8_negative.py", "starter/slot_ledger.py"
            )
            $expectedDependencies = [ordered]@{
                "starter/attributes.py" = $AttributesOracleBlobOid
                "starter/p8_negative.py" = $P8NegativeOracleBlobOid
                "starter/slot_ledger.py" = $SlotLedgerOracleBlobOid
            }
            foreach ($relative in $expectedDependencies.Keys) {
                $actual = $outer.target_receipt.dependency_blob_oids.PSObject.Properties[[string]$relative].Value
                if ($actual -isnot [string] -or [string]$actual -cne [string]$expectedDependencies[$relative]) {
                    Throw-Code "PRECLAIM_ORACLE_DEPENDENCY"
                }
            }
            if (
                -not ($outer.target_receipt.comparison_count -is [int] -or $outer.target_receipt.comparison_count -is [long]) -or
                [int64]$outer.target_receipt.comparison_count -ne $OracleComparisonCount -or
                -not ($outer.target_receipt.fixture_count -is [int] -or $outer.target_receipt.fixture_count -is [long]) -or
                [int64]$outer.target_receipt.fixture_count -ne $OracleFixtureCount -or
                $outer.target_receipt.algorithm_fts_cache_privacy_pass -isnot [bool] -or
                $outer.target_receipt.algorithm_fts_cache_privacy_pass -ne $true -or
                $outer.target_receipt.exact_repeat -isnot [bool] -or $outer.target_receipt.exact_repeat -ne $true -or
                $outer.target_receipt.isolated_namespace -isnot [bool] -or $outer.target_receipt.isolated_namespace -ne $true -or
                $outer.target_receipt.legacy_runtime_absent -isnot [bool] -or $outer.target_receipt.legacy_runtime_absent -ne $true -or
                $outer.target_receipt.matrix_sha256 -isnot [string] -or
                [string]$outer.target_receipt.matrix_sha256 -cne $OracleMatrixSha256 -or
                $outer.target_receipt.parent_blob_oid -isnot [string] -or
                [string]$outer.target_receipt.parent_blob_oid -cne $ParentOracleBlobOid -or
                $outer.target_receipt.status -isnot [string] -or
                [string]$outer.target_receipt.status -cne "ORACLE_DIFFERENTIAL_SELF_CHECK_PASS"
            ) { Throw-Code "PRECLAIM_ORACLE_RECEIPT" }
            $receipts += $outer.target_receipt
        }
        finally {
            if ($null -ne $checkRuntime -and [System.IO.Directory]::Exists([string]$checkRuntime.Root)) {
                Remove-OwnedFreshRuntime -Runtime $checkRuntime
            }
        }
    }
    if ($receipts.Count -ne 2) { Throw-Code "PRECLAIM_ORACLE_REPEAT" }
    Assert-CanonicalObjectEqual -Left $receipts[0] -Right $receipts[1] -Code "PRECLAIM_ORACLE_REPEAT"
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
            "--target-module", "scripts.probe_oov_chargram_bridge_g0",
            "--target-blob", [string]$Blobs[$RunnerRelative],
            "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
            "--",
            "--preclaim-chain-self-check", "--implementation-commit", $Commit
        )
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment (New-OfflineEnvironment -TempPath $checkRuntime.Temp) -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_PREREQUISITE_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedMode "direct" -ExpectedPycache $checkRuntime.Pycache
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

function Invoke-PreclaimCleanupLifecycleCheck {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct", "module")][string]$InvocationMode,
        [Parameter(Mandatory = $true)]$Blobs
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
            $modulePath = Join-Path $moduleRoot "v223_safe_bootstrap.py"
            $bootstrapBytes = Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608
            Write-ExclusiveBytes -Path $modulePath -Bytes $bootstrapBytes
            $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
            $arguments += @("-m", "v223_safe_bootstrap")
        }
        $arguments += @(
            "--mode", $InvocationMode,
            "--target-path", $RunnerPath.Replace("\", "/"),
            "--target-module", "scripts.probe_oov_chargram_bridge_g0",
            "--target-blob", [string]$Blobs[$RunnerRelative],
            "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
            "--", "--runtime-cleanup-self-check"
        )
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_CLEANUP_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedMode $InvocationMode -ExpectedPycache $checkRuntime.Pycache
        Assert-ExactNames -Value $outer.target_receipt -Expected @(
            "create_write_close_publish_cleanup", "mode", "source_only", "status"
        )
        if (
            $capture.ExitCode -ne 0 -or [int64]$outer.target_exit_code -ne 0 -or
            $outer.target_receipt.create_write_close_publish_cleanup -ne $true -or
            $outer.target_receipt.source_only -ne $true -or
            [string]$outer.target_receipt.mode -cne $InvocationMode -or
            [string]$outer.target_receipt.status -cne "RUNTIME_CLEANUP_SELF_CHECK_PASS"
        ) { Throw-Code "PRECLAIM_CLEANUP_RECEIPT" }
    }
    finally {
        if ($null -ne $checkRuntime -and [System.IO.Directory]::Exists([string]$checkRuntime.Root)) {
            Remove-OwnedFreshRuntime -Runtime $checkRuntime
        }
    }
}

function Invoke-PreclaimCleanupLifecycleChecks {
    param([Parameter(Mandatory = $true)]$Blobs)
    foreach ($invocationMode in @("direct", "module")) {
        Invoke-PreclaimCleanupLifecycleCheck -InvocationMode $invocationMode -Blobs $Blobs
    }
}

function Invoke-PreclaimFailurePropagationCheck {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("direct", "module")][string]$InvocationMode,
        [Parameter(Mandatory = $true)]$Blobs
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
            $modulePath = Join-Path $moduleRoot "v223_safe_bootstrap.py"
            Write-ExclusiveBytes -Path $modulePath -Bytes (Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608)
            $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
            $arguments += @("-m", "v223_safe_bootstrap")
        }
        $arguments += @(
            "--mode", $InvocationMode,
            "--target-path", $RunnerPath.Replace("\", "/"),
            "--target-module", "scripts.probe_oov_chargram_bridge_g0",
            "--target-blob", [string]$Blobs[$RunnerRelative],
            "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
            "--", "--failure-propagation-self-check"
        )
        $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
        if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
            Throw-Code "PRECLAIM_FAILURE_ENVELOPE"
        }
        $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedMode $InvocationMode -ExpectedPycache $checkRuntime.Pycache
        if ($capture.ExitCode -ne 0 -or [int64]$outer.target_exit_code -ne 0 -or $null -eq $outer.target_receipt) {
            Throw-Code "PRECLAIM_FAILURE_EXIT"
        }
        Assert-ExactNames -Value $outer.target_receipt -Expected @(
            "direct_or_module_source_only", "identifier_injection_rejected",
            "malformed_and_extra_keys_rejected", "root_code_preserved", "status",
            "stderr_payload_discarded"
        )
        if (
            $outer.target_receipt.direct_or_module_source_only -ne $true -or
            $outer.target_receipt.identifier_injection_rejected -ne $true -or
            $outer.target_receipt.malformed_and_extra_keys_rejected -ne $true -or
            $outer.target_receipt.root_code_preserved -ne $true -or
            $outer.target_receipt.stderr_payload_discarded -ne $true -or
            [string]$outer.target_receipt.status -cne "FAILURE_PROPAGATION_SELF_CHECK_PASS"
        ) { Throw-Code "PRECLAIM_FAILURE_RECEIPT" }
    }
    finally {
        if ($null -ne $checkRuntime -and [System.IO.Directory]::Exists([string]$checkRuntime.Root)) {
            Remove-OwnedFreshRuntime -Runtime $checkRuntime
        }
    }
}

function Invoke-PreclaimLegacyRuntimeDenialChecks {
    param([Parameter(Mandatory = $true)]$Blobs)
    foreach ($invocationMode in @("direct", "module")) {
        $checkRuntime = $null
        try {
            $checkRuntime = New-FreshRuntime
            $environment = New-OfflineEnvironment -TempPath $checkRuntime.Temp
            $arguments = @("-P", "-S", "-s", "-B", "-X", "pycache_prefix=$($checkRuntime.Pycache.Replace('\','/'))")
            if ($invocationMode -eq "direct") {
                $arguments += $BootstrapPath.Replace("\", "/")
            }
            else {
                $moduleRoot = Join-Path $checkRuntime.Root "module"
                [void][IO.Directory]::CreateDirectory($moduleRoot)
                Write-ExclusiveBytes -Path (Join-Path $moduleRoot "v223_safe_bootstrap.py") -Bytes (Get-PlainFileBytes -Path $BootstrapPath -MaximumBytes 8388608)
                $environment["PYTHONPATH"] = $moduleRoot.Replace("\", "/")
                $arguments += @("-m", "v223_safe_bootstrap")
            }
            $arguments += @(
                "--mode", $invocationMode,
                "--target-path", $WorkerPath.Replace("\", "/"),
                "--target-module", "scripts.oov_chargram_bridge_g0_worker",
                "--target-blob", [string]$Blobs[$WorkerRelative],
                "--bootstrap-blob", [string]$Blobs[$BootstrapRelative],
                "--", "--entrypoint-self-check", "--require-module",
                "starter.sparse_multiview_g0", "--stage-id", "preclaim_synthetic"
            )
            $capture = Invoke-CapturedProcess -FileName $PythonExe -Arguments $arguments -WorkingDirectory $ProjectRoot -Environment $environment -TimeoutMilliseconds 60000
            if (-not (Test-BasicOuterEnvelope -Stdout $capture.Stdout -Stderr $capture.Stderr -TimedOut $capture.TimedOut)) {
                Throw-Code "PRECLAIM_LEGACY_DENIAL_ENVELOPE"
            }
            $outer = Parse-And-ValidateOuter -Raw $capture.Stdout -ExpectedRunnerBlob ([string]$Blobs[$WorkerRelative]) -ExpectedBootstrapBlob ([string]$Blobs[$BootstrapRelative]) -ExpectedMode $invocationMode -ExpectedPycache $checkRuntime.Pycache
            if ($capture.ExitCode -eq 0 -or [int64]$outer.target_exit_code -eq 0 -or $null -ne $outer.target_receipt) {
                Throw-Code "PRECLAIM_LEGACY_RUNTIME_NOT_DENIED"
            }
        }
        finally {
            if ($null -ne $checkRuntime -and [IO.Directory]::Exists([string]$checkRuntime.Root)) {
                Remove-OwnedFreshRuntime -Runtime $checkRuntime
            }
        }
    }
}

function Invoke-PreclaimFailurePropagationChecks {
    param([Parameter(Mandatory = $true)]$Blobs)
    foreach ($invocationMode in @("direct", "module")) {
        Invoke-PreclaimFailurePropagationCheck -InvocationMode $invocationMode -Blobs $Blobs
    }
}

function Assert-FormalGitReceipt {
    param(
        [Parameter(Mandatory = $true)]$Git,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)]$Blobs
    )
    Assert-ExactNames -Value $Git -Expected @(
        "branch", "commit", "implementation_blobs", "object_only_git",
        "preregistration_commit", "pushed", "remote"
    )
    Assert-ExactNames -Value $Git.implementation_blobs -Expected $ImplementationPaths
    if (
        [string]$Git.branch -cne $Branch -or [string]$Git.commit -cne $Commit -or
        [string]$Git.preregistration_commit -cne $PreregCommit -or
        [string]$Git.remote -cne $RemoteUrl -or $Git.pushed -isnot [bool] -or
        $Git.pushed -ne $true -or $Git.object_only_git -isnot [bool] -or
        $Git.object_only_git -ne $true
    ) { Throw-Code "FORMAL_RECEIPT_GIT" }
    foreach ($relative in $ImplementationPaths) {
        if ([string]$Git.implementation_blobs.PSObject.Properties[$relative].Value -cne [string]$Blobs[$relative]) {
            Throw-Code "FORMAL_RECEIPT_SOURCE_BLOBS"
        }
    }
}

function Assert-FormalCommonReceipt {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)]$Blobs,
        [Parameter(Mandatory = $true)]$ClaimIdentity,
        [Parameter(Mandatory = $true)]$Bootstrap
    )
    if (
        [string]$Receipt.schema_version -cne $ProbeSchemaVersion -or
        [string]$Receipt.experiment_id -cne "SR-V2.23-TARGET-BLIND-OOV-CHARGRAM-LEXICON-BRIDGE-G0" -or
        [string]$Receipt.mode -cne $Mode -or [string]$Receipt.recorded_on -cne $RecordedOn -or
        $Receipt.rerun_forbidden -isnot [bool] -or $Receipt.rerun_forbidden -ne $true
    ) { Throw-Code "FORMAL_RECEIPT_IDENTITY" }
    Assert-IdentityEqual -Observed $Receipt.claim -Expected $ClaimIdentity -Code "FORMAL_RECEIPT_CLAIM_BINDING"
    Assert-CanonicalObjectEqual -Left $Receipt.bootstrap -Right $Bootstrap -Code "FORMAL_RECEIPT_BOOTSTRAP_BINDING"
    Assert-ExactNames -Value $Receipt.preregistration -Expected @("bytes", "rows", "sha256")
    if (
        -not ($Receipt.preregistration.bytes -is [int] -or $Receipt.preregistration.bytes -is [long]) -or
        [int64]$Receipt.preregistration.bytes -ne $PreregBytes -or
        [string]$Receipt.preregistration.sha256 -cne $PreregSha256 -or
        -not ($Receipt.preregistration.rows -is [int] -or $Receipt.preregistration.rows -is [long]) -or
        [int64]$Receipt.preregistration.rows -le 0
    ) { Throw-Code "FORMAL_RECEIPT_PREREGISTRATION" }
    Assert-ExactNames -Value $Receipt.runtime -Expected @(
        "cpu_only", "critical_binary_dependencies", "git", "git_sha256",
        "gpu_peak_bytes", "python", "python_sha256", "sqlite"
    )
    if (
        $Receipt.runtime.cpu_only -isnot [bool] -or $Receipt.runtime.cpu_only -ne $true -or
        [int64]$Receipt.runtime.gpu_peak_bytes -ne 0 -or
        [string]$Receipt.runtime.python -cne "3.11.16" -or
        [string]$Receipt.runtime.python_sha256 -cne $PythonSha256 -or
        [string]$Receipt.runtime.sqlite -cne "3.53.4" -or
        [string]$Receipt.runtime.git -cne $GitVersion -or
        [string]$Receipt.runtime.git_sha256 -cne $GitSha256
    ) { Throw-Code "FORMAL_RECEIPT_RUNTIME" }
    Assert-ExactNames -Value $Receipt.runtime.critical_binary_dependencies -Expected @(
        "D:/450/conda/envs/tiktok/python311.dll",
        "D:/450/conda/envs/tiktok/DLLs/_sqlite3.pyd",
        "D:/450/conda/envs/tiktok/Library/bin/sqlite3.dll"
    )
    foreach ($binary in $CriticalBinaries) {
        $key = ([string]$binary.Path).Replace("\", "/")
        $observed = $Receipt.runtime.critical_binary_dependencies.PSObject.Properties[$key].Value
        Assert-ExactNames -Value $observed -Expected @("bytes", "rows", "sha256")
        Assert-ExactInteger -Value $observed.bytes -Expected ([int64]$binary.Bytes) -Code "FORMAL_RECEIPT_CRITICAL_BINARY"
        if (
            -not ($observed.rows -is [int] -or $observed.rows -is [long]) -or
            [int64]$observed.rows -lt 0 -or
            [string]$observed.sha256 -cne [string]$binary.Sha256
        ) { Throw-Code "FORMAL_RECEIPT_CRITICAL_BINARY" }
    }
    Assert-FormalGitReceipt -Git $Receipt.git -Commit $Commit -Blobs $Blobs
}

function Get-FiniteDouble {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Code
    )
    if (
        $Value -is [bool] -or
        -not (
            $Value -is [int] -or $Value -is [long] -or
            $Value -is [double] -or $Value -is [decimal]
        )
    ) { Throw-Code $Code }
    $number = [double]$Value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { Throw-Code $Code }
    return $number
}

function Assert-FrozenSourceIdentity {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][int64]$Bytes,
        [Parameter(Mandatory = $true)][string]$Sha256,
        [int64]$Rows = -1,
        [Parameter(Mandatory = $true)][string]$Code
    )
    if ($Rows -ge 0) {
        Assert-ExactNames -Value $Value -Expected @("bytes", "rows", "sha256")
        Assert-ExactInteger -Value $Value.rows -Expected $Rows -Code $Code
    }
    else {
        Assert-ExactNames -Value $Value -Expected @("bytes", "sha256")
    }
    Assert-ExactInteger -Value $Value.bytes -Expected $Bytes -Code $Code
    if ([string]$Value.sha256 -cne $Sha256) { Throw-Code $Code }
}

function Assert-FormalSources {
    param(
        [Parameter(Mandatory = $true)]$Sources,
        [Parameter(Mandatory = $true)][bool]$IncludeTargets
    )
    $expectedNames = @("catalog", "sealed_c200", "visible_context")
    if ($IncludeTargets) { $expectedNames += @("numeric_label_archive", "proxy") }
    Assert-ExactNames -Value $Sources -Expected $expectedNames
    Assert-FrozenSourceIdentity -Value $Sources.catalog -Bytes 60546327L -Rows 50000L -Sha256 "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67" -Code "FORMAL_SOURCE_CATALOG"
    Assert-FrozenSourceIdentity -Value $Sources.visible_context -Bytes 47168882L -Rows 2000L -Sha256 "f30a98700da5d480731fe7e82c87c40a22f06de290e069e20dc68f9fefecd20f" -Code "FORMAL_SOURCE_CONTEXT"
    $sealed = @($Sources.sealed_c200)
    if ($sealed.Count -ne 2) { Throw-Code "FORMAL_SOURCE_C200" }
    foreach ($replica in $sealed) {
        Assert-FrozenSourceIdentity -Value $replica -Bytes 32226135L -Rows 20000L -Sha256 "a8589749376f48f019997a618481578dde36be4ca1fc723e8ed00056c23e40dc" -Code "FORMAL_SOURCE_C200"
    }
    if (-not $IncludeTargets) { return }
    Assert-FrozenSourceIdentity -Value $Sources.proxy -Bytes 1315338L -Rows 2000L -Sha256 "2175696171c0d874fca4b9aa456ff5fd7d570f2184f59ade6781198f6443198e" -Code "FORMAL_SOURCE_PROXY"
    Assert-ExactNames -Value $Sources.numeric_label_archive -Expected @("bytes", "members_read_in_order", "sha256")
    Assert-ExactInteger -Value $Sources.numeric_label_archive.bytes -Expected 1702876L -Code "FORMAL_SOURCE_LABEL"
    if ([string]$Sources.numeric_label_archive.sha256 -cne "9cf8f76e88fa386cfe32cb0e262e6ffd0738ac90676473065cb7d1e4dfcc48eb") {
        Throw-Code "FORMAL_SOURCE_LABEL"
    }
    $members = @($Sources.numeric_label_archive.members_read_in_order)
    $expectedMembers = @("outer_fold", "family_index", "taxonomy_code")
    if ($members.Count -ne $expectedMembers.Count) { Throw-Code "FORMAL_SOURCE_LABEL_MEMBERS" }
    for ($index = 0; $index -lt $expectedMembers.Count; $index++) {
        if ($members[$index] -isnot [string] -or [string]$members[$index] -cne $expectedMembers[$index]) {
            Throw-Code "FORMAL_SOURCE_LABEL_MEMBERS"
        }
    }
}

function Assert-FormalCacheSnapshot {
    param(
        [Parameter(Mandatory = $true)]$Snapshot,
        [Parameter(Mandatory = $true)][bool]$AfterClose,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-ExactNames -Value $Snapshot -Expected @(
        "fts_route", "mask_decision", "oov_bridge", "product_view"
    )
    $capacities = [ordered]@{ oov_bridge = 4096L; fts_route = 512L; product_view = 4096L; mask_decision = 16384L }
    foreach ($layerName in $capacities.Keys) {
        $layer = $Snapshot.PSObject.Properties[$layerName].Value
        Assert-ExactNames -Value $layer -Expected @(
            "capacity", "closed", "evictions", "hits", "inserts", "misses", "size"
        )
        Assert-ExactBoolean -Value $layer.closed -Expected $AfterClose -Code $Code
        foreach ($field in @("capacity", "evictions", "hits", "inserts", "misses", "size")) {
            $number = $layer.PSObject.Properties[$field].Value
            if (-not ($number -is [int] -or $number -is [long]) -or [int64]$number -lt 0) { Throw-Code $Code }
        }
        if (
            [int64]$layer.capacity -ne [int64]$capacities[$layerName] -or
            [int64]$layer.size -gt [int64]$layer.capacity -or
            [int64]$layer.inserts -gt [int64]$layer.misses -or
            [int64]$layer.evictions -gt [int64]$layer.inserts -or
            ($AfterClose -and [int64]$layer.size -ne 0)
        ) { Throw-Code $Code }
    }
}

function Assert-FormalCachePair {
    param(
        [Parameter(Mandatory = $true)]$Cache,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-ExactNames -Value $Cache -Expected @("after_close", "before_close")
    Assert-FormalCacheSnapshot -Snapshot $Cache.before_close -AfterClose $false -Code $Code
    Assert-FormalCacheSnapshot -Snapshot $Cache.after_close -AfterClose $true -Code $Code
    foreach ($layerName in @("oov_bridge", "fts_route", "product_view", "mask_decision")) {
        $before = $Cache.before_close.PSObject.Properties[$layerName].Value
        $after = $Cache.after_close.PSObject.Properties[$layerName].Value
        foreach ($field in @("capacity", "evictions", "hits", "inserts", "misses")) {
            if ([int64]$before.PSObject.Properties[$field].Value -ne [int64]$after.PSObject.Properties[$field].Value) {
                Throw-Code $Code
            }
        }
    }
}

function Assert-FormalLatencySummary {
    param(
        [Parameter(Mandatory = $true)]$Summary,
        [Parameter(Mandatory = $true)][int64]$ExpectedCount,
        [Parameter(Mandatory = $true)][double]$MaximumP95,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-ExactNames -Value $Summary -Expected @(
        "count", "maximum_milliseconds", "p50_milliseconds", "p95_milliseconds"
    )
    Assert-ExactInteger -Value $Summary.count -Expected $ExpectedCount -Code $Code
    $maximum = Get-FiniteDouble -Value $Summary.maximum_milliseconds -Code $Code
    $p50 = Get-FiniteDouble -Value $Summary.p50_milliseconds -Code $Code
    $p95 = Get-FiniteDouble -Value $Summary.p95_milliseconds -Code $Code
    if (
        $maximum -lt 0.0 -or $p50 -lt 0.0 -or $p95 -lt 0.0 -or
        $p50 -gt $p95 -or $p95 -gt $maximum -or $p95 -gt $MaximumP95 -or
        ($ExpectedCount -eq 0 -and ($maximum -ne 0.0 -or $p50 -ne 0.0 -or $p95 -ne 0.0))
    ) { Throw-Code $Code }
    return $p95
}

function Assert-FormalCompactWorker {
    param(
        [Parameter(Mandatory = $true)]$Worker,
        [Parameter(Mandatory = $true)][ValidateSet("direct", "module")][string]$ExpectedMode,
        [Parameter(Mandatory = $true)][bool]$ExpectedCache,
        [Parameter(Mandatory = $true)][int64]$ExpectedSessions,
        [Parameter(Mandatory = $true)][bool]$FullRun,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $expectedNames = @(
        "activation", "cache_enabled", "child_wall_seconds", "device", "gpu_peak_bytes",
        "latency", "mode", "network_attempt_count", "parent_peak_working_set_bytes",
        "parent_wall_seconds", "peak_working_set_bytes", "provider", "semantic_trace",
        "trace", "turns_per_second"
    )
    if ($ExpectedCache) { $expectedNames += "cache" }
    Assert-ExactNames -Value $Worker -Expected $expectedNames
    if ([string]$Worker.mode -cne $ExpectedMode) { Throw-Code $Code }
    Assert-ExactBoolean -Value $Worker.cache_enabled -Expected $ExpectedCache -Code $Code
    Assert-ExactInteger -Value $Worker.network_attempt_count -Expected 0L -Code $Code
    Assert-ExactInteger -Value $Worker.gpu_peak_bytes -Expected 0L -Code $Code
    if ([string]$Worker.device -cne "CPU" -or [string]$Worker.provider -cnotmatch "^SQLite FTS5") { Throw-Code $Code }

    $parentWall = Get-FiniteDouble -Value $Worker.parent_wall_seconds -Code $Code
    $childWall = Get-FiniteDouble -Value $Worker.child_wall_seconds -Code $Code
    $throughput = Get-FiniteDouble -Value $Worker.turns_per_second -Code $Code
    if (
        $parentWall -le 0.0 -or ($FullRun -and $parentWall -gt 1800.0) -or
        $childWall -le 0.0 -or $childWall -gt 1800.0 -or
        $throughput -le 0.0 -or ($FullRun -and $throughput -lt 10.0)
    ) { Throw-Code $Code }
    foreach ($field in @("parent_peak_working_set_bytes", "peak_working_set_bytes")) {
        $value = $Worker.PSObject.Properties[$field].Value
        if (-not ($value -is [int] -or $value -is [long]) -or [int64]$value -le 0 -or [int64]$value -gt 1610612736L) {
            Throw-Code $Code
        }
    }
    if ([int64]$Worker.parent_peak_working_set_bytes -lt [int64]$Worker.peak_working_set_bytes) { Throw-Code $Code }

    $records = $ExpectedSessions * 10L
    Assert-ExactNames -Value $Worker.semantic_trace -Expected @("rows", "sha256")
    Assert-ExactInteger -Value $Worker.semantic_trace.rows -Expected $records -Code $Code
    if ([string]$Worker.semantic_trace.sha256 -cnotmatch "^[0-9a-f]{64}$") { Throw-Code $Code }

    Assert-ExactNames -Value $Worker.trace -Expected @(
        "bytes", "c200_cells", "candidate_cell_ratio_over_c200", "candidate_cells",
        "expansion_sessions", "expansion_turns", "max_candidates", "min_candidates",
        "rows", "sha256", "trace_byte_ratio_over_c200"
    )
    foreach ($field in @("bytes", "c200_cells", "candidate_cells", "expansion_sessions", "expansion_turns", "max_candidates", "min_candidates", "rows")) {
        $value = $Worker.trace.PSObject.Properties[$field].Value
        if (-not ($value -is [int] -or $value -is [long]) -or [int64]$value -lt 0) { Throw-Code $Code }
    }
    Assert-ExactInteger -Value $Worker.trace.rows -Expected $records -Code $Code
    if (
        [int64]$Worker.trace.bytes -le 0 -or [int64]$Worker.trace.c200_cells -le 0 -or
        [int64]$Worker.trace.candidate_cells -lt [int64]$Worker.trace.c200_cells -or
        [int64]$Worker.trace.min_candidates -gt [int64]$Worker.trace.max_candidates -or
        [int64]$Worker.trace.max_candidates -gt 400 -or
        [int64]$Worker.trace.expansion_sessions -gt $ExpectedSessions -or
        [int64]$Worker.trace.expansion_turns -gt $records -or
        [string]$Worker.trace.sha256 -cnotmatch "^[0-9a-f]{64}$"
    ) { Throw-Code $Code }
    $cellRatio = Get-FiniteDouble -Value $Worker.trace.candidate_cell_ratio_over_c200 -Code $Code
    $traceRatio = Get-FiniteDouble -Value $Worker.trace.trace_byte_ratio_over_c200 -Code $Code
    $expectedCellRatio = [Math]::Round(([double]$Worker.trace.candidate_cells / [double]$Worker.trace.c200_cells), 6, [MidpointRounding]::ToEven)
    if (
        [Math]::Abs($cellRatio - $expectedCellRatio) -gt 0.0000005 -or
        $cellRatio -lt 1.0 -or $traceRatio -le 0.0 -or
        ($FullRun -and ($cellRatio -gt 2.0 -or $traceRatio -gt 2.1))
    ) { Throw-Code $Code }

    Assert-ExactNames -Value $Worker.activation -Expected @(
        "bridge_mapping_records", "fts_route_records", "legacy_route_executions",
        "novel_candidate_cells", "v222_route_executions", "valid_oov_source_records"
    )
    foreach ($property in $Worker.activation.PSObject.Properties) {
        if (-not ($property.Value -is [int] -or $property.Value -is [long]) -or [int64]$property.Value -lt 0) { Throw-Code $Code }
    }
    Assert-ExactInteger -Value $Worker.activation.legacy_route_executions -Expected 0L -Code $Code
    Assert-ExactInteger -Value $Worker.activation.v222_route_executions -Expected 0L -Code $Code
    # Source extraction/mapping precedes the six selected-source cap.  Only the
    # mapped bridge and its one-route-per-source executions are capped at six.
    foreach ($field in @("bridge_mapping_records", "fts_route_records")) {
        if ([int64]$Worker.activation.PSObject.Properties[$field].Value -gt ($records * 6L)) { Throw-Code $Code }
    }

    Assert-ExactNames -Value $Worker.latency -Expected @(
        "bridge_lookup", "context_container_parse", "extra_bridge_and_mask", "fts_route",
        "hard_conflict_mask", "per_turn"
    )
    # Context parsing is required in the receipt but has no preregistered P95
    # threshold.  Validate its exact count/shape/finiteness without inventing
    # a post-hoc resource gate.
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.context_container_parse -ExpectedCount $ExpectedSessions -MaximumP95 ([double]::MaxValue) -Code $Code
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.extra_bridge_and_mask -ExpectedCount $records -MaximumP95 100.0 -Code $Code
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.hard_conflict_mask -ExpectedCount $records -MaximumP95 50.0 -Code $Code
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.per_turn -ExpectedCount $records -MaximumP95 400.0 -Code $Code
    $bridgeCountValue = $Worker.latency.bridge_lookup.count
    $routeCountValue = $Worker.latency.fts_route.count
    if (
        -not ($bridgeCountValue -is [int] -or $bridgeCountValue -is [long]) -or
        -not ($routeCountValue -is [int] -or $routeCountValue -is [long])
    ) { Throw-Code $Code }
    $bridgeCount = [int64]$bridgeCountValue
    $routeCount = [int64]$routeCountValue
    if ($bridgeCount -lt 0 -or $bridgeCount -gt ($records * 6L) -or $routeCount -lt 0 -or $routeCount -gt ($records * 6L)) { Throw-Code $Code }
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.bridge_lookup -ExpectedCount $bridgeCount -MaximumP95 25.0 -Code $Code
    $null = Assert-FormalLatencySummary -Summary $Worker.latency.fts_route -ExpectedCount $routeCount -MaximumP95 25.0 -Code $Code
    if ($ExpectedCache) {
        Assert-FormalCachePair -Cache $Worker.cache -Code $Code
        if ($routeCount -ne [int64]$Worker.cache.before_close.fts_route.misses) { Throw-Code $Code }
    }
    return [pscustomobject]@{
        Activation = $Worker.activation
        Cache = $(if ($ExpectedCache) { $Worker.cache } else { $null })
        ParentWall = $parentWall
        Semantic = $Worker.semantic_trace
        Trace = $Worker.trace
    }
}

function Assert-FormalPreflightStage {
    param(
        [Parameter(Mandatory = $true)]$Stage,
        [Parameter(Mandatory = $true)][ValidateSet(20, 100)][int]$ExpectedSessions,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-ExactNames -Value $Stage -Expected @(
        "cached_pair_parent_wall_seconds", "exact_triplet", "information_available",
        "linear_extrapolation_x1_5_seconds", "no_information_reasons",
        "session_limit", "stage_wall_seconds", "workers"
    )
    Assert-ExactInteger -Value $Stage.session_limit -Expected ([int64]$ExpectedSessions) -Code $Code
    Assert-ExactBoolean -Value $Stage.exact_triplet -Expected $true -Code $Code
    if ($Stage.information_available -isnot [bool]) { Throw-Code $Code }
    $cachedPair = Get-FiniteDouble -Value $Stage.cached_pair_parent_wall_seconds -Code $Code
    $extrapolation = Get-FiniteDouble -Value $Stage.linear_extrapolation_x1_5_seconds -Code $Code
    $stageWall = Get-FiniteDouble -Value $Stage.stage_wall_seconds -Code $Code
    if ($cachedPair -le 0.0 -or $extrapolation -le 0.0 -or $stageWall -le 0.0) { Throw-Code $Code }
    if ($ExpectedSessions -eq 100 -and ($cachedPair -gt 60.0 -or $extrapolation -gt 1800.0)) { Throw-Code $Code }

    $workers = @($Stage.workers)
    if ($workers.Count -ne 3) { Throw-Code $Code }
    $facts = @(
        (Assert-FormalCompactWorker -Worker $workers[0] -ExpectedMode direct -ExpectedCache $false -ExpectedSessions $ExpectedSessions -FullRun $false -Code $Code),
        (Assert-FormalCompactWorker -Worker $workers[1] -ExpectedMode direct -ExpectedCache $true -ExpectedSessions $ExpectedSessions -FullRun $false -Code $Code),
        (Assert-FormalCompactWorker -Worker $workers[2] -ExpectedMode module -ExpectedCache $true -ExpectedSessions $ExpectedSessions -FullRun $false -Code $Code)
    )
    foreach ($index in 1..2) {
        Assert-CanonicalObjectEqual -Left $facts[0].Trace -Right $facts[$index].Trace -Code $Code
        Assert-CanonicalObjectEqual -Left $facts[0].Semantic -Right $facts[$index].Semantic -Code $Code
        Assert-CanonicalObjectEqual -Left $facts[0].Activation -Right $facts[$index].Activation -Code $Code
    }
    $expectedPair = [Math]::Round(($facts[1].ParentWall + $facts[2].ParentWall), 6, [MidpointRounding]::ToEven)
    $expectedExtrapolation = [Math]::Round(
        ([Math]::Max($facts[1].ParentWall, $facts[2].ParentWall) * (2000.0 / $ExpectedSessions) * 1.5),
        6,
        [MidpointRounding]::ToEven
    )
    # The stage receipt was computed from unrounded clocks, while compact
    # worker clocks are already rounded to six decimals.  Bound only the
    # maximum possible quantization propagation; do not demand an impossible
    # exact reverse calculation.
    $extrapolationTolerance = ((2000.0 / $ExpectedSessions) * 1.5 * 0.0000005) + 0.000001
    if (
        [Math]::Abs($cachedPair - $expectedPair) -gt 0.000002 -or
        [Math]::Abs($extrapolation - $expectedExtrapolation) -gt $extrapolationTolerance -or
        ($stageWall + 0.000005) -lt ($facts[0].ParentWall + $facts[1].ParentWall + $facts[2].ParentWall)
    ) { Throw-Code $Code }

    $expectedReasons = New-Object System.Collections.Generic.List[string]
    if ($ExpectedSessions -eq 100) {
        foreach ($field in @("valid_oov_source_records", "bridge_mapping_records", "fts_route_records", "novel_candidate_cells")) {
            if ([int64]$facts[0].Activation.PSObject.Properties[$field].Value -le 0) { $expectedReasons.Add("${field}_zero") }
        }
        for ($index = 1; $index -le 2; $index++) {
            $modeName = if ($index -eq 1) { "direct" } else { "module" }
            $cacheHits = 0L
            foreach ($layer in @("oov_bridge", "fts_route", "product_view", "mask_decision")) {
                $cacheHits += [int64]$facts[$index].Cache.before_close.PSObject.Properties[$layer].Value.hits
            }
            if ($cacheHits -le 0) { $expectedReasons.Add("${modeName}_cache_hits_zero") }
        }
    }
    $expectedReasonArray = @($expectedReasons | Sort-Object -Unique)
    $observedReasons = @($Stage.no_information_reasons)
    if ($observedReasons.Count -ne $expectedReasonArray.Count) { Throw-Code $Code }
    for ($index = 0; $index -lt $expectedReasonArray.Count; $index++) {
        if ($observedReasons[$index] -isnot [string] -or [string]$observedReasons[$index] -cne [string]$expectedReasonArray[$index]) {
            Throw-Code $Code
        }
    }
    $expectedInformation = $expectedReasonArray.Count -eq 0
    if ($Stage.information_available -ne $expectedInformation) { Throw-Code $Code }
    return [pscustomobject]@{
        InformationAvailable = $expectedInformation
        Workers = $workers
        WorkerFacts = $facts
    }
}

function Assert-RecallViewMap {
    param(
        [Parameter(Mandatory = $true)]$Views,
        [Parameter(Mandatory = $true)][int64]$Denominator,
        [Parameter(Mandatory = $true)][string]$Code
    )
    $viewOrder = @(
        "SEALED_K10", "SEALED_K20", "SEALED_K50", "SEALED_K100",
        "SEALED_VARIABLE_C200", "EXPANDED_FIXED_K200", "C400_COMPLETE_UNION"
    )
    Assert-ExactNames -Value $Views -Expected $viewOrder
    $counts = [ordered]@{}
    foreach ($view in $viewOrder) {
        $entry = $Views.PSObject.Properties[$view].Value
        Assert-ExactNames -Value $entry -Expected @("count", "fraction")
        $countValue = $entry.count
        if (
            -not ($countValue -is [int] -or $countValue -is [long]) -or
            [int64]$countValue -lt 0 -or [int64]$countValue -gt $Denominator
        ) { Throw-Code $Code }
        $fraction = Get-FiniteDouble -Value $entry.fraction -Code $Code
        $expectedFraction = if ($Denominator -eq 0) { 0.0 } else {
            [Math]::Round(([double]$countValue / [double]$Denominator), 6, [MidpointRounding]::ToEven)
        }
        if ($fraction -lt 0.0 -or $fraction -gt 1.0 -or [Math]::Abs($fraction - $expectedFraction) -gt 0.0000005) {
            Throw-Code $Code
        }
        $counts[$view] = [int64]$countValue
    }
    if (
        $counts.SEALED_K10 -gt $counts.SEALED_K20 -or
        $counts.SEALED_K20 -gt $counts.SEALED_K50 -or
        $counts.SEALED_K50 -gt $counts.SEALED_K100 -or
        $counts.SEALED_K100 -gt $counts.SEALED_VARIABLE_C200 -or
        $counts.SEALED_VARIABLE_C200 -gt $counts.EXPANDED_FIXED_K200 -or
        $counts.EXPANDED_FIXED_K200 -gt $counts.C400_COMPLETE_UNION
    ) { Throw-Code $Code }
    return $counts
}

function Assert-FormalCandidateRecall {
    param(
        [Parameter(Mandatory = $true)]$Recall,
        [Parameter(Mandatory = $true)][string]$Code
    )
    Assert-ExactNames -Value $Recall -Expected @(
        "all_2000_sessions", "by_outer_fold", "by_taxonomy", "c200_absent_frontier",
        "exact_target_cluster_uniform", "family_disjoint_audit", "increment"
    )
    $global = Assert-RecallViewMap -Views $Recall.all_2000_sessions -Denominator 2000L -Code $Code
    $baselineExpected = [ordered]@{
        SEALED_K10 = 1895L
        SEALED_K20 = 1943L
        SEALED_K50 = 1982L
        SEALED_K100 = 1986L
        SEALED_VARIABLE_C200 = 1986L
    }
    foreach ($key in $baselineExpected.Keys) {
        if ([int64]$global[$key] -ne [int64]$baselineExpected[$key]) { Throw-Code $Code }
    }

    Assert-ExactNames -Value $Recall.c200_absent_frontier -Expected @("sessions", "views")
    Assert-ExactInteger -Value $Recall.c200_absent_frontier.sessions -Expected 14L -Code $Code
    $frontier = Assert-RecallViewMap -Views $Recall.c200_absent_frontier.views -Denominator 14L -Code $Code
    if ([int64]$frontier.SEALED_VARIABLE_C200 -ne 0) { Throw-Code $Code }

    Assert-ExactNames -Value $Recall.increment -Expected @(
        "count", "non_clothing_count", "outer_fold_span", "target_cluster_count", "taxonomy_span"
    )
    foreach ($field in @("count", "non_clothing_count", "outer_fold_span", "target_cluster_count", "taxonomy_span")) {
        $value = $Recall.increment.PSObject.Properties[$field].Value
        if (-not ($value -is [int] -or $value -is [long]) -or [int64]$value -lt 0) { Throw-Code $Code }
    }
    $increment = [int64]$Recall.increment.count
    if (
        $increment -gt 14 -or
        $increment -ne ([int64]$global.C400_COMPLETE_UNION - 1986L) -or
        $increment -ne [int64]$frontier.C400_COMPLETE_UNION -or
        [int64]$Recall.increment.non_clothing_count -gt $increment -or
        [int64]$Recall.increment.outer_fold_span -gt 5 -or
        [int64]$Recall.increment.taxonomy_span -gt 4 -or
        [int64]$Recall.increment.target_cluster_count -gt $increment
    ) { Throw-Code $Code }

    $foldRows = @($Recall.by_outer_fold)
    if ($foldRows.Count -ne 5) { Throw-Code $Code }
    $foldSessionTotal = 0L
    $foldIncrementTotal = 0L
    $foldSpan = 0L
    $foldViewTotals = [ordered]@{}
    foreach ($view in $global.Keys) { $foldViewTotals[$view] = 0L }
    for ($fold = 0; $fold -lt 5; $fold++) {
        $row = $foldRows[$fold]
        Assert-ExactNames -Value $row -Expected @("fold", "increment", "sessions", "views")
        Assert-ExactInteger -Value $row.fold -Expected ([int64]$fold) -Code $Code
        foreach ($field in @("sessions", "increment")) {
            $value = $row.PSObject.Properties[$field].Value
            if (-not ($value -is [int] -or $value -is [long]) -or [int64]$value -lt 0) { Throw-Code $Code }
        }
        if ([int64]$row.increment -gt [int64]$row.sessions) { Throw-Code $Code }
        $counts = Assert-RecallViewMap -Views $row.views -Denominator ([int64]$row.sessions) -Code $Code
        if ([int64]$row.increment -ne ([int64]$counts.C400_COMPLETE_UNION - [int64]$counts.SEALED_VARIABLE_C200)) { Throw-Code $Code }
        $foldSessionTotal += [int64]$row.sessions
        $foldIncrementTotal += [int64]$row.increment
        if ([int64]$row.increment -gt 0) { $foldSpan++ }
        foreach ($view in $global.Keys) { $foldViewTotals[$view] += [int64]$counts[$view] }
    }
    if ($foldSessionTotal -ne 2000L -or $foldIncrementTotal -ne $increment -or $foldSpan -ne [int64]$Recall.increment.outer_fold_span) { Throw-Code $Code }
    foreach ($view in $global.Keys) {
        if ([int64]$foldViewTotals[$view] -ne [int64]$global[$view]) { Throw-Code $Code }
    }

    $taxonomyNames = @("accessories-other", "clothing", "jewelry", "shoes")
    Assert-ExactNames -Value $Recall.by_taxonomy -Expected $taxonomyNames
    $taxonomySessionTotal = 0L
    $taxonomyIncrementTotal = 0L
    $taxonomySpan = 0L
    $nonClothing = 0L
    $taxonomyViewTotals = [ordered]@{}
    foreach ($view in $global.Keys) { $taxonomyViewTotals[$view] = 0L }
    foreach ($taxonomy in $taxonomyNames) {
        $row = $Recall.by_taxonomy.PSObject.Properties[$taxonomy].Value
        Assert-ExactNames -Value $row -Expected @("increment", "sessions", "views")
        foreach ($field in @("sessions", "increment")) {
            $value = $row.PSObject.Properties[$field].Value
            if (-not ($value -is [int] -or $value -is [long]) -or [int64]$value -lt 0) { Throw-Code $Code }
        }
        if ([int64]$row.increment -gt [int64]$row.sessions) { Throw-Code $Code }
        $counts = Assert-RecallViewMap -Views $row.views -Denominator ([int64]$row.sessions) -Code $Code
        if ([int64]$row.increment -ne ([int64]$counts.C400_COMPLETE_UNION - [int64]$counts.SEALED_VARIABLE_C200)) { Throw-Code $Code }
        $taxonomySessionTotal += [int64]$row.sessions
        $taxonomyIncrementTotal += [int64]$row.increment
        if ([int64]$row.increment -gt 0) { $taxonomySpan++ }
        if ($taxonomy -cne "clothing") { $nonClothing += [int64]$row.increment }
        foreach ($view in $global.Keys) { $taxonomyViewTotals[$view] += [int64]$counts[$view] }
    }
    if (
        $taxonomySessionTotal -ne 2000L -or $taxonomyIncrementTotal -ne $increment -or
        $taxonomySpan -ne [int64]$Recall.increment.taxonomy_span -or
        $nonClothing -ne [int64]$Recall.increment.non_clothing_count
    ) { Throw-Code $Code }
    foreach ($view in $global.Keys) {
        if ([int64]$taxonomyViewTotals[$view] -ne [int64]$global[$view]) { Throw-Code $Code }
    }

    Assert-ExactNames -Value $Recall.exact_target_cluster_uniform -Expected @(
        "c400_complete_union_fraction", "cluster_count", "delta", "sealed_variable_c200_fraction"
    )
    $clusterCountValue = $Recall.exact_target_cluster_uniform.cluster_count
    if (-not ($clusterCountValue -is [int] -or $clusterCountValue -is [long]) -or [int64]$clusterCountValue -le 0 -or [int64]$clusterCountValue -gt 2000) { Throw-Code $Code }
    $baselineUniform = Get-FiniteDouble -Value $Recall.exact_target_cluster_uniform.sealed_variable_c200_fraction -Code $Code
    $candidateUniform = Get-FiniteDouble -Value $Recall.exact_target_cluster_uniform.c400_complete_union_fraction -Code $Code
    $uniformDelta = Get-FiniteDouble -Value $Recall.exact_target_cluster_uniform.delta -Code $Code
    $expectedDelta = [Math]::Round(($candidateUniform - $baselineUniform), 9, [MidpointRounding]::ToEven)
    if (
        $baselineUniform -lt 0.0 -or $baselineUniform -gt 1.0 -or
        $candidateUniform -lt $baselineUniform -or $candidateUniform -gt 1.0 -or
        # The runner rounds baseline, candidate and their unrounded difference
        # independently to nine decimals.  Allow only that bounded double-
        # rounding envelope when checking the reported delta relation.
        [Math]::Abs($uniformDelta - $expectedDelta) -gt 0.0000000016
    ) { Throw-Code $Code }

    Assert-ExactNames -Value $Recall.family_disjoint_audit -Expected @(
        "families_crossing_outer_folds", "family_count", "valid"
    )
    Assert-ExactBoolean -Value $Recall.family_disjoint_audit.valid -Expected $true -Code $Code
    Assert-ExactInteger -Value $Recall.family_disjoint_audit.families_crossing_outer_folds -Expected 0L -Code $Code
    $familyCount = $Recall.family_disjoint_audit.family_count
    if (-not ($familyCount -is [int] -or $familyCount -is [long]) -or [int64]$familyCount -le 0 -or [int64]$familyCount -gt 2000) { Throw-Code $Code }

    return [pscustomobject]@{
        C400 = [int64]$global.C400_COMPLETE_UNION
        Increment = $increment
        NonClothing = [int64]$Recall.increment.non_clothing_count
        OuterFoldSpan = [int64]$Recall.increment.outer_fold_span
        UniformDelta = $uniformDelta
    }
}

function Assert-FormalTargetReceipt {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)]$Blobs,
        [Parameter(Mandatory = $true)]$ClaimIdentity,
        [Parameter(Mandatory = $true)]$Bootstrap,
        $ExpectedPrerequisite
    )
    if ($Mode -eq "preflight") {
        Assert-ExactNames -Value $Receipt -Expected @(
            "bootstrap", "claim", "device", "entrypoint_regression", "experiment_id", "git",
            "implementation", "integrity", "mode", "next", "preregistration", "recorded_on",
            "resources", "rerun_forbidden", "runtime", "schema_version", "sources", "stages", "status"
        )
        Assert-FormalCommonReceipt -Receipt $Receipt -Commit $Commit -Blobs $Blobs -ClaimIdentity $ClaimIdentity -Bootstrap $Bootstrap
        if ([string]$Receipt.status -cnotin @("TARGET_FREE_PREFLIGHT_COMPLETE", "PRE_OUTCOME_NO_INFORMATION")) {
            Throw-Code "FORMAL_PREFLIGHT_STATUS"
        }
        Assert-ExactNames -Value $Receipt.implementation -Expected @(
            "branch", "commit", "default_off", "preregistration_commit",
            "served_top10_unchanged", "target_blind"
        )
        if (
            [string]$Receipt.implementation.branch -cne $Branch -or
            [string]$Receipt.implementation.commit -cne $Commit -or
            [string]$Receipt.implementation.preregistration_commit -cne $PreregCommit
        ) { Throw-Code "FORMAL_PREFLIGHT_IMPLEMENTATION" }
        Assert-ExactBoolean -Value $Receipt.implementation.default_off -Expected $true -Code "FORMAL_PREFLIGHT_IMPLEMENTATION"
        Assert-ExactBoolean -Value $Receipt.implementation.served_top10_unchanged -Expected $true -Code "FORMAL_PREFLIGHT_IMPLEMENTATION"
        Assert-ExactBoolean -Value $Receipt.implementation.target_blind -Expected $true -Code "FORMAL_PREFLIGHT_IMPLEMENTATION"
        Assert-ExactNames -Value $Receipt.integrity -Expected @(
            "exact_triplet_each_stage", "legacy_route_executions", "network_attempt_count",
            "ordered_variable_c200_prefix", "target_sources_opened"
        )
        Assert-ExactBoolean -Value $Receipt.integrity.exact_triplet_each_stage -Expected $true -Code "FORMAL_PREFLIGHT_INTEGRITY"
        Assert-ExactBoolean -Value $Receipt.integrity.ordered_variable_c200_prefix -Expected $true -Code "FORMAL_PREFLIGHT_INTEGRITY"
        Assert-ExactBoolean -Value $Receipt.integrity.target_sources_opened -Expected $false -Code "FORMAL_PREFLIGHT_INTEGRITY"
        Assert-ExactInteger -Value $Receipt.integrity.legacy_route_executions -Expected 0L -Code "FORMAL_PREFLIGHT_INTEGRITY"
        Assert-ExactInteger -Value $Receipt.integrity.network_attempt_count -Expected 0L -Code "FORMAL_PREFLIGHT_INTEGRITY"
        Assert-ExactNames -Value $Receipt.device -Expected @("gpu_peak_bytes", "reason", "selected")
        Assert-ExactInteger -Value $Receipt.device.gpu_peak_bytes -Expected 0L -Code "FORMAL_PREFLIGHT_DEVICE"
        if (
            [string]$Receipt.device.selected -cne "CPU" -or
            [string]$Receipt.device.reason -cne "frozen SQLite FTS5 chargram/edit-distance bridge backend"
        ) { Throw-Code "FORMAL_PREFLIGHT_DEVICE" }
        Assert-ExactNames -Value $Receipt.entrypoint_regression -Expected @(
            "legacy_module_denied_direct", "legacy_module_denied_module", "runner_direct",
            "runner_module", "worker_direct", "worker_module"
        )
        foreach ($property in $Receipt.entrypoint_regression.PSObject.Properties) {
            Assert-ExactBoolean -Value $property.Value -Expected $true -Code "FORMAL_PREFLIGHT_ENTRYPOINT"
        }
        Assert-FormalSources -Sources $Receipt.sources -IncludeTargets $false
        Assert-ExactNames -Value $Receipt.resources -Expected @(
            "free_disk_bytes_before_formal", "limit100_cached_direct_plus_module_parent_wall_seconds",
            "limit100_linear_extrapolation_x1_5_seconds"
        )
        if (-not ($Receipt.resources.free_disk_bytes_before_formal -is [int] -or $Receipt.resources.free_disk_bytes_before_formal -is [long]) -or [int64]$Receipt.resources.free_disk_bytes_before_formal -lt 536870912L) {
            Throw-Code "FORMAL_PREFLIGHT_RESOURCE"
        }
        $pairWall = Get-FiniteDouble -Value $Receipt.resources.limit100_cached_direct_plus_module_parent_wall_seconds -Code "FORMAL_PREFLIGHT_RESOURCE"
        $fullEstimate = Get-FiniteDouble -Value $Receipt.resources.limit100_linear_extrapolation_x1_5_seconds -Code "FORMAL_PREFLIGHT_RESOURCE"
        if ($pairWall -le 0.0 -or $pairWall -gt 60.0 -or $fullEstimate -le 0.0 -or $fullEstimate -gt 1800.0) { Throw-Code "FORMAL_PREFLIGHT_RESOURCE" }
        $stages = @($Receipt.stages)
        if ($stages.Count -ne 2) { Throw-Code "FORMAL_PREFLIGHT_STAGE" }
        $stage20 = Assert-FormalPreflightStage -Stage $stages[0] -ExpectedSessions 20 -Code "FORMAL_PREFLIGHT_STAGE"
        $stage100 = Assert-FormalPreflightStage -Stage $stages[1] -ExpectedSessions 100 -Code "FORMAL_PREFLIGHT_STAGE"
        if (
            [Math]::Abs($pairWall - (Get-FiniteDouble -Value $stages[1].cached_pair_parent_wall_seconds -Code "FORMAL_PREFLIGHT_RESOURCE")) -gt 0.0000005 -or
            [Math]::Abs($fullEstimate - (Get-FiniteDouble -Value $stages[1].linear_extrapolation_x1_5_seconds -Code "FORMAL_PREFLIGHT_RESOURCE")) -gt 0.0000005
        ) { Throw-Code "FORMAL_PREFLIGHT_RESOURCE_BINDING" }
        $hasInformation = $stage100.InformationAvailable
        if (([string]$Receipt.status -ceq "TARGET_FREE_PREFLIGHT_COMPLETE") -ne $hasInformation) {
            Throw-Code "FORMAL_PREFLIGHT_INFORMATION_STATUS"
        }
        $expectedNext = if ($hasInformation) {
            "one immutable full-2000 candidate-recall receipt"
        }
        else {
            "freeze v2.23 without candidate claim and choose an independent mechanism"
        }
        if ($Receipt.next -isnot [string] -or [string]$Receipt.next -cne $expectedNext) { Throw-Code "FORMAL_PREFLIGHT_NEXT" }
        return
    }

    Assert-ExactNames -Value $Receipt -Expected @(
        "baseline_sanity", "bootstrap", "candidate_recall", "candidate_retention", "claim",
        "decision", "evidence_scope", "exact_repeat", "experiment_id", "git", "implementation",
        "mode", "preflight_prerequisite", "preregistration", "recorded_on", "rerun_forbidden",
        "resources", "runtime", "schema_version", "sources", "status"
    )
    Assert-FormalCommonReceipt -Receipt $Receipt -Commit $Commit -Blobs $Blobs -ClaimIdentity $ClaimIdentity -Bootstrap $Bootstrap
    if ([string]$Receipt.status -cnotin @(
        "CANDIDATE_RECALL_GO_ALLOW_SEPARATE_POLICY_PREREGISTRATION",
        "CANDIDATE_RECALL_NO_GO_FREEZE_EXACT_G0"
    )) { Throw-Code "FORMAL_CANDIDATE_STATUS" }
    if (
        $Receipt.evidence_scope -isnot [string] -or
        [string]$Receipt.evidence_scope -cne "shared 2000-session diagnostic candidate membership; not served HR@10 and not private validation"
    ) { Throw-Code "FORMAL_CANDIDATE_EVIDENCE_SCOPE" }
    Assert-CanonicalObjectEqual -Left $Receipt.preflight_prerequisite -Right $ExpectedPrerequisite -Code "FORMAL_CANDIDATE_PREREQUISITE"
    Assert-FormalSources -Sources $Receipt.sources -IncludeTargets $true
    Assert-ExactNames -Value $Receipt.baseline_sanity -Expected @(
        "SEALED_K10", "SEALED_K20", "SEALED_K50", "SEALED_K100", "SEALED_VARIABLE_C200"
    )
    $baselineExpected = [ordered]@{
        SEALED_K10 = 1895L
        SEALED_K20 = 1943L
        SEALED_K50 = 1982L
        SEALED_K100 = 1986L
        SEALED_VARIABLE_C200 = 1986L
    }
    foreach ($key in $baselineExpected.Keys) {
        Assert-ExactInteger -Value $Receipt.baseline_sanity.PSObject.Properties[$key].Value -Expected ([int64]$baselineExpected[$key]) -Code "FORMAL_CANDIDATE_BASELINE"
    }
    $recallFacts = Assert-FormalCandidateRecall -Recall $Receipt.candidate_recall -Code "FORMAL_CANDIDATE_RECALL"
    Assert-ExactNames -Value $Receipt.implementation -Expected @(
        "branch", "commit", "default_off", "full_agent_evaluator_started",
        "preregistration_commit", "runtime_target_blind", "served_top10_unchanged"
    )
    if (
        [string]$Receipt.implementation.branch -cne $Branch -or
        [string]$Receipt.implementation.commit -cne $Commit -or
        [string]$Receipt.implementation.preregistration_commit -cne $PreregCommit
    ) { Throw-Code "FORMAL_CANDIDATE_IMPLEMENTATION" }
    Assert-ExactBoolean -Value $Receipt.implementation.default_off -Expected $true -Code "FORMAL_CANDIDATE_IMPLEMENTATION"
    Assert-ExactBoolean -Value $Receipt.implementation.runtime_target_blind -Expected $true -Code "FORMAL_CANDIDATE_IMPLEMENTATION"
    Assert-ExactBoolean -Value $Receipt.implementation.served_top10_unchanged -Expected $true -Code "FORMAL_CANDIDATE_IMPLEMENTATION"
    Assert-ExactBoolean -Value $Receipt.implementation.full_agent_evaluator_started -Expected $false -Code "FORMAL_CANDIDATE_IMPLEMENTATION"
    Assert-ExactNames -Value $Receipt.exact_repeat -Expected @(
        "passed", "semantic_sha256", "trace_bytes", "trace_rows", "trace_sha256"
    )
    Assert-ExactBoolean -Value $Receipt.exact_repeat.passed -Expected $true -Code "FORMAL_CANDIDATE_REPEAT"
    Assert-ExactInteger -Value $Receipt.exact_repeat.trace_rows -Expected 20000L -Code "FORMAL_CANDIDATE_REPEAT"
    if (
        -not ($Receipt.exact_repeat.trace_bytes -is [int] -or $Receipt.exact_repeat.trace_bytes -is [long]) -or
        [int64]$Receipt.exact_repeat.trace_bytes -le 0 -or
        [string]$Receipt.exact_repeat.trace_sha256 -cnotmatch "^[0-9a-f]{64}$" -or
        [string]$Receipt.exact_repeat.semantic_sha256 -cnotmatch "^[0-9a-f]{64}$"
    ) { Throw-Code "FORMAL_CANDIDATE_REPEAT" }
    Assert-ExactNames -Value $Receipt.candidate_retention -Expected @(
        "c200_duplicate_count", "c200_loss_count", "c200_reorder_count",
        "complete_variable_c200_exact_ordered_prefix", "served_top10_unchanged",
        "tail_duplicate_count", "tail_explicit_hard_conflict_count"
    )
    Assert-ExactBoolean -Value $Receipt.candidate_retention.complete_variable_c200_exact_ordered_prefix -Expected $true -Code "FORMAL_CANDIDATE_RETENTION"
    Assert-ExactBoolean -Value $Receipt.candidate_retention.served_top10_unchanged -Expected $true -Code "FORMAL_CANDIDATE_RETENTION"
    foreach ($field in @("c200_duplicate_count", "c200_loss_count", "c200_reorder_count", "tail_duplicate_count", "tail_explicit_hard_conflict_count")) {
        Assert-ExactInteger -Value $Receipt.candidate_retention.PSObject.Properties[$field].Value -Expected 0L -Code "FORMAL_CANDIDATE_RETENTION"
    }
    Assert-ExactNames -Value $Receipt.resources -Expected @(
        "budgets_passed", "free_disk_bytes_before_formal", "full_formal_parent_wall_seconds",
        "gpu_peak_bytes", "network_attempt_count", "pre_target_parent_wall_seconds", "workers"
    )
    $workers = @($Receipt.resources.workers)
    Assert-ExactBoolean -Value $Receipt.resources.budgets_passed -Expected $true -Code "FORMAL_CANDIDATE_RESOURCE"
    Assert-ExactInteger -Value $Receipt.resources.network_attempt_count -Expected 0L -Code "FORMAL_CANDIDATE_RESOURCE"
    Assert-ExactInteger -Value $Receipt.resources.gpu_peak_bytes -Expected 0L -Code "FORMAL_CANDIDATE_RESOURCE"
    if (-not ($Receipt.resources.free_disk_bytes_before_formal -is [int] -or $Receipt.resources.free_disk_bytes_before_formal -is [long]) -or [int64]$Receipt.resources.free_disk_bytes_before_formal -lt 536870912L -or $workers.Count -ne 2) {
        Throw-Code "FORMAL_CANDIDATE_RESOURCE"
    }
    $preTargetWall = Get-FiniteDouble -Value $Receipt.resources.pre_target_parent_wall_seconds -Code "FORMAL_CANDIDATE_RESOURCE"
    $fullFormalWall = Get-FiniteDouble -Value $Receipt.resources.full_formal_parent_wall_seconds -Code "FORMAL_CANDIDATE_RESOURCE"
    if ($preTargetWall -le 0.0 -or $preTargetWall -gt 1800.0 -or $fullFormalWall -lt $preTargetWall -or $fullFormalWall -gt 1800.0) {
        Throw-Code "FORMAL_CANDIDATE_RESOURCE"
    }
    $workerFacts = @(
        (Assert-FormalCompactWorker -Worker $workers[0] -ExpectedMode direct -ExpectedCache $true -ExpectedSessions 2000L -FullRun $true -Code "FORMAL_CANDIDATE_WORKER"),
        (Assert-FormalCompactWorker -Worker $workers[1] -ExpectedMode module -ExpectedCache $true -ExpectedSessions 2000L -FullRun $true -Code "FORMAL_CANDIDATE_WORKER")
    )
    Assert-CanonicalObjectEqual -Left $workerFacts[0].Trace -Right $workerFacts[1].Trace -Code "FORMAL_CANDIDATE_REPEAT"
    Assert-CanonicalObjectEqual -Left $workerFacts[0].Semantic -Right $workerFacts[1].Semantic -Code "FORMAL_CANDIDATE_REPEAT"
    Assert-CanonicalObjectEqual -Left $workerFacts[0].Activation -Right $workerFacts[1].Activation -Code "FORMAL_CANDIDATE_REPEAT"
    if (
        $preTargetWall + 0.000005 -lt ($workerFacts[0].ParentWall + $workerFacts[1].ParentWall) -or
        [int64]$Receipt.exact_repeat.trace_bytes -ne [int64]$workerFacts[0].Trace.bytes -or
        [string]$Receipt.exact_repeat.trace_sha256 -cne [string]$workerFacts[0].Trace.sha256 -or
        [string]$Receipt.exact_repeat.semantic_sha256 -cne [string]$workerFacts[0].Semantic.sha256
    ) { Throw-Code "FORMAL_CANDIDATE_RESOURCE_BINDING" }
    Assert-ExactNames -Value $Receipt.decision -Expected @(
        "candidate_threshold", "fallback_order", "next_stage", "promotion_gate_passed", "top10_global_promotion"
    )
    $goStatus = [string]$Receipt.status -ceq "CANDIDATE_RECALL_GO_ALLOW_SEPARATE_POLICY_PREREGISTRATION"
    $computedGo = (
        $recallFacts.C400 -ge 1988L -and $recallFacts.Increment -ge 2L -and
        $recallFacts.OuterFoldSpan -ge 2L -and $recallFacts.NonClothing -ge 1L -and
        $recallFacts.UniformDelta -gt 0.0
    )
    $fallback = @($Receipt.decision.fallback_order)
    $expectedFallback = @("SR-V2.12-FIXED-TWO-PAGE-GRACE", "v1.9", "P11", "R08")
    if ($fallback.Count -ne $expectedFallback.Count) { Throw-Code "FORMAL_CANDIDATE_DECISION" }
    for ($index = 0; $index -lt $expectedFallback.Count; $index++) {
        if ($fallback[$index] -isnot [string] -or [string]$fallback[$index] -cne $expectedFallback[$index]) { Throw-Code "FORMAL_CANDIDATE_DECISION" }
    }
    $expectedNextStage = if ($computedGo) {
        "separate preregistration for the 100-session policy smoke"
    }
    else {
        "freeze this exact G0 and choose the next independent mechanism"
    }
    if (
        $Receipt.decision.promotion_gate_passed -isnot [bool] -or
        $Receipt.decision.promotion_gate_passed -ne $computedGo -or
        $goStatus -ne $computedGo -or
        -not ($Receipt.decision.candidate_threshold -is [int] -or $Receipt.decision.candidate_threshold -is [long]) -or
        [int64]$Receipt.decision.candidate_threshold -ne 1988L -or
        $Receipt.decision.top10_global_promotion -isnot [bool] -or
        $Receipt.decision.top10_global_promotion -ne $false -or
        $Receipt.decision.next_stage -isnot [string] -or [string]$Receipt.decision.next_stage -cne $expectedNextStage
    ) { Throw-Code "FORMAL_CANDIDATE_DECISION" }
}

function Get-OuterTransactionReportBytes {
    param(
        [Parameter(Mandatory = $true)][byte[]]$TerminalBytes,
        [Parameter(Mandatory = $true)][System.Diagnostics.Stopwatch]$Stopwatch,
        [Parameter(Mandatory = $true)][string]$TerminalStatus
    )
    if ($Stopwatch.IsRunning) { $Stopwatch.Stop() }
    if ($Stopwatch.Elapsed.TotalSeconds -le 0.0) { Throw-Code "OUTER_TRANSACTION_CLOCK" }
    return Get-CanonicalBytes -Value ([ordered]@{
        durable_terminal = [ordered]@{
            bytes = $TerminalBytes.Length
            sha256 = (Get-Sha256Hex -Bytes $TerminalBytes)
            status = $TerminalStatus
        }
        outer_transaction_wall_seconds = [double]$Stopwatch.Elapsed.TotalSeconds
        schema_version = "small-ranker-v2.23-outer-transaction-report.v1"
        status = "OUTER_TRANSACTION_RECORDED"
    })
}

$claimed = $false
$outerWritten = $false
$terminalWritten = $false
$attemptPaths = $null
$runtime = $null
$sourceBlobs = $null
$preflightPrerequisite = $null
$claimIdentity = $null
$processCapture = $null
$outerRaw = $null
$outerIsRawEnvelope = $false
$runtimeCleanupPassed = $false
$formalReceiptValidated = $false
$validatedFailure = $null
$parsedOuter = $null
$finalBytes = $null
$finalExitCode = 2
$outerStopwatch = New-Object System.Diagnostics.Stopwatch

try {
    Assert-FormalPowerShellHost -RequestedMode $Mode -RequestedCommit $ImplementationCommit
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
    Invoke-PreclaimLegacyRuntimeDenialChecks -Blobs $sourceBlobs
    Invoke-PreclaimCleanupLifecycleChecks -Blobs $sourceBlobs
    Invoke-PreclaimFailurePropagationChecks -Blobs $sourceBlobs
    Invoke-PreclaimOracleDifferentialChecks -Blobs $sourceBlobs
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
        experiment_id = "SR-V2.23-TARGET-BLIND-OOV-CHARGRAM-LEXICON-BRIDGE-G0"
        implementation_commit = $ImplementationCommit
        mode = $Mode
        one_shot = $true
        preregistration = [ordered]@{ blob = $PreregBlob; commit = $PreregCommit }
        preregistration_commit = $PreregCommit
        recorded_on = $RecordedOn
        schema_version = "small-ranker-v2.23-durable-one-shot-claim.v1"
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
    try {
        Write-ExclusiveBytes -Path ([string]$attemptPaths["claim"]) -Bytes $claimBytes -CreationObserved ([ref]$claimed)
        if (-not $claimed) { Throw-Code "CLAIM_CREATE_STATE" }
        $claimIdentity = Get-RawIdentity -Raw $claimBytes

        $runtime = New-FreshRuntime
        $pycachePosix = $runtime.Pycache.Replace("\", "/")
        $arguments = @(
            "-P", "-S", "-s", "-B", "-X", "pycache_prefix=$pycachePosix",
            $BootstrapPath.Replace("\", "/"),
            "--mode", "direct",
            "--target-path", $RunnerPath.Replace("\", "/"),
            "--target-module", "scripts.probe_oov_chargram_bridge_g0",
            "--target-blob", [string]$sourceBlobs[$RunnerRelative],
            "--bootstrap-blob", [string]$sourceBlobs[$BootstrapRelative],
            "--",
            "--run", "--mode", $Mode, "--implementation-commit", $ImplementationCommit
        )
        $outerStopwatch.Restart()
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

        $validatedFailure = $null
        if ($outerIsRawEnvelope) {
            $parsedOuter = Parse-And-ValidateOuter -Raw $outerRaw -ExpectedRunnerBlob ([string]$sourceBlobs[$RunnerRelative]) -ExpectedBootstrapBlob ([string]$sourceBlobs[$BootstrapRelative]) -ExpectedMode "direct" -ExpectedPycache $runtime.Pycache
            if ([int64]$parsedOuter.target_exit_code -eq 0) {
                if ($processCapture.ExitCode -ne 0 -or $null -eq $parsedOuter.target_receipt) {
                    Throw-Code "OUTER_SUCCESS_EXIT_DIVERGENCE"
                }
                Assert-FormalTargetReceipt -Receipt $parsedOuter.target_receipt -Commit $ImplementationCommit -Blobs $sourceBlobs -ClaimIdentity $claimIdentity -Bootstrap $parsedOuter.bootstrap -ExpectedPrerequisite $preflightPrerequisite
                $formalReceiptValidated = $true
            }
            elseif ($null -ne $parsedOuter.target_receipt) {
                $validatedFailure = Assert-RunnerFailureReceipt -Receipt $parsedOuter.target_receipt
                $formalReceiptValidated = $true
            }
        }
        Remove-OwnedFreshRuntime -Runtime $runtime
        $runtimeCleanupPassed = $true
        $runtime = $null
        $complete = $outerIsRawEnvelope -and -not $processCapture.TimedOut -and $processCapture.ExitCode -eq 0 -and $null -ne $parsedOuter -and [int]$parsedOuter.target_exit_code -eq 0 -and $null -ne $parsedOuter.target_receipt -and $runtimeCleanupPassed -and $formalReceiptValidated
        if ($complete) {
            $terminal = [ordered]@{
                implementation_commit = $ImplementationCommit
                mode = $Mode
                outer = [ordered]@{ bytes = $outerRaw.Length; sha256 = (Get-Sha256Hex -Bytes $outerRaw) }
                preregistration = [ordered]@{ blob = $PreregBlob; commit = $PreregCommit }
                process_exit_code = [int]$processCapture.ExitCode
                raw_stderr_retained = $false
                recorded_on = $RecordedOn
                schema_version = "small-ranker-v2.23-durable-terminal.v1"
                status = "COMPLETE"
                target_exit_code = [int]$parsedOuter.target_exit_code
                target_receipt = $parsedOuter.target_receipt
            }
        }
        elseif ($null -ne $validatedFailure) {
            $rootCode = if ([string]$validatedFailure.failure_origin -ceq "worker") { [string]$validatedFailure.worker_error_code } else { [string]$validatedFailure.runner_error_code }
            $terminal = New-InvalidTerminal -PowerShellCode "RUNNER_FAILURE" -RootFailureOrigin ([string]$validatedFailure.failure_origin) -RootStageId ([string]$validatedFailure.stage_id) -RootErrorCode $rootCode -CanonicalFailureReceipt ([bool]$validatedFailure.canonical_failure_receipt)
        }
        elseif ($outerIsRawEnvelope) {
            $terminal = New-InvalidTerminal -PowerShellCode "BOOTSTRAP_FAILURE" -RootFailureOrigin "bootstrap"
        }
        else {
            $terminal = New-InvalidTerminal -PowerShellCode "OUTER_CAPTURE_FAILURE"
        }
        Assert-ReceiptPrivacy -Value $terminal
        $finalBytes = Get-CanonicalBytes -Value $terminal
        if ($finalBytes.Length -gt $MaximumTerminalBytes) { Throw-Code "TERMINAL_SIZE_GATE" }
        Write-ExclusiveBytes -Path ([string]$attemptPaths["result"]) -Bytes $finalBytes
        $terminalWritten = $true
        $durableTerminalBytes = $finalBytes
        $finalBytes = Get-OuterTransactionReportBytes -TerminalBytes $durableTerminalBytes -Stopwatch $outerStopwatch -TerminalStatus ([string]$terminal.status)
        $finalExitCode = $(if ($complete) { 0 } else { 2 })
    }
    catch {
        # FileMode.CreateNew, not the later verification/hash, is the one-shot
        # boundary.  Once this process created the fixed claim path, every
        # failure must finish the durable INVALID chain.
        if (-not $claimed) { throw }
        $code = Get-SafeErrorCode -Caught $_
        if ($null -ne $runtime -and [System.IO.Directory]::Exists([string]$runtime.Root)) {
            try {
                Remove-OwnedFreshRuntime -Runtime $runtime
                $runtimeCleanupPassed = $true
                $runtime = $null
            }
            catch {
                $code = Get-SafeErrorCode -Caught $_
            }
        }
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
            if ($null -ne $validatedFailure) {
                $rootCode = if ([string]$validatedFailure.failure_origin -ceq "worker") { [string]$validatedFailure.worker_error_code } else { [string]$validatedFailure.runner_error_code }
                $invalid = New-InvalidTerminal -PowerShellCode "TRANSACTION_FAILURE" -RootFailureOrigin ([string]$validatedFailure.failure_origin) -RootStageId ([string]$validatedFailure.stage_id) -RootErrorCode $rootCode -CanonicalFailureReceipt ([bool]$validatedFailure.canonical_failure_receipt)
            }
            else {
                $invalid = New-InvalidTerminal -PowerShellCode "TRANSACTION_FAILURE" -RootFailureOrigin "powershell"
            }
            try {
                Assert-ReceiptPrivacy -Value $invalid
                $finalBytes = Get-CanonicalBytes -Value $invalid
                Write-ExclusiveBytes -Path ([string]$attemptPaths["result"]) -Bytes $finalBytes
                $terminalWritten = $true
                $durableTerminalBytes = $finalBytes
                $finalBytes = Get-OuterTransactionReportBytes -TerminalBytes $durableTerminalBytes -Stopwatch $outerStopwatch -TerminalStatus "INVALID_ONE_SHOT_CONSUMED"
            }
            catch {
                $finalBytes = Get-CanonicalBytes -Value ([ordered]@{
                    error_code = "TERMINAL_DURABLE_WRITE_FAILED"
                    implementation_commit = $ImplementationCommit
                    mode = $Mode
                    recorded_on = $RecordedOn
                    schema_version = "small-ranker-v2.23-ephemeral-failure.v1"
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
        schema_version = "small-ranker-v2.23-ephemeral-failure.v1"
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
        schema_version = "small-ranker-v2.23-ephemeral-failure.v1"
        status = $(if ($claimed) { "ONE_SHOT_CONSUMED_CRASH" } else { "PRELAUNCH_BLOCKED_NOT_CONSUMED" })
    })
    $finalExitCode = 2
}

Write-ConsoleCanonical -Bytes $finalBytes
exit $finalExitCode
