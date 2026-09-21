#requires -Version 5.1
[CmdletBinding()]
param(
    [string]$DestinationRoot = 'D:\AAA_MyNote\FNO_backup\FNO_Kerr_CrossResolution',
    [ValidateRange(1, 3)][int]$MaxAttempts = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$SshAlias = 'server-school'
$ManifestPath = Join-Path $PSScriptRoot 'MENTOR_PACKAGE_DOWNLOAD_MANIFEST.tsv'
$ExpectedManifestHash = '04b412b921c2899c4d711f388908e81341bd10d32093088186f1295aa35f3d79'
$ExpectedCount = 218
$ExpectedBytes = [long]1905335335

if ($env:OS -ne 'Windows_NT') { throw 'Run this script on Windows only.' }
$Scp = (Get-Command scp.exe -CommandType Application -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw 'Place the reviewed download TSV beside this script.'
}
if ((Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash -ne $ExpectedManifestHash) {
    throw 'Download manifest hash mismatch. No download was started.'
}
$Rows = @(Import-Csv -LiteralPath $ManifestPath -Delimiter "`t" -Encoding UTF8)
$Root = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')
$RootPrefix = $Root + '\'
$SeenSources = @{}
$SeenTargets = @{}
$Items = @()
[long]$TotalBytes = 0

# 下载前一次性验证清单，禁止路径穿越、重复目标和非项目源路径。
foreach ($Row in $Rows) {
    $Source = [string]$Row.server_absolute_path
    $Relative = [string]$Row.windows_relative_path
    if ($Source -notmatch '^/home/shanjinshuo/fno_kerr/kerr_project/[A-Za-z0-9_./-]+$' -or
        $Source -match '(^|/)\.\.(/|$)') { throw 'Invalid server source path.' }
    if ($Relative -notmatch '^[A-Za-z0-9_./-]+$' -or $Relative.StartsWith('/') -or
        $Relative -match '(^|/)\.\.?(/|$)') { throw 'Invalid Windows relative path.' }
    if ($Row.size_bytes -notmatch '^\d+$') { throw 'Invalid expected file size.' }
    $Hash = [string]$Row.sha256_if_known
    if ($Hash -and $Hash -ne 'unknown' -and $Hash -notmatch '^[0-9a-fA-F]{64}$') {
        throw 'Invalid SHA256 value.'
    }
    $Target = [IO.Path]::GetFullPath((Join-Path $Root $Relative.Replace('/', '\')))
    if (-not $Target.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Target escapes the package directory.'
    }
    if ($SeenSources.ContainsKey($Source) -or $SeenTargets.ContainsKey($Target)) {
        throw 'Duplicate source or target in manifest.'
    }
    $SeenSources[$Source] = $true
    $SeenTargets[$Target] = $true
    $TotalBytes += [long]$Row.size_bytes
    $Items += [pscustomobject]@{ Row = $Row; Target = $Target }
}
if ($Items.Count -ne $ExpectedCount -or $TotalBytes -ne $ExpectedBytes) {
    throw 'Package count or size differs from the reviewed scope.'
}

# 不沿本地符号链接或目录联接写入目标目录之外的位置。
function Assert-SafeLocalPath([string]$Path) {
    $Cursor = [IO.Path]::GetFullPath($Path)
    while ($Cursor) {
        if (Test-Path -LiteralPath $Cursor) {
            $Info = Get-Item -LiteralPath $Cursor -Force
            if (($Info.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw 'Local reparse points are not supported.'
            }
        }
        $Parent = [IO.Directory]::GetParent($Cursor)
        if ($null -eq $Parent) { break }
        $Cursor = $Parent.FullName
    }
}

function Test-VerifiedFile([string]$Path, $Row) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Row.size_bytes) { return $false }
    if ($Row.sha256_if_known -and $Row.sha256_if_known -ne 'unknown') {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Row.sha256_if_known
    }
    return $true
}

Assert-SafeLocalPath $Root
[IO.Directory]::CreateDirectory($Root) | Out-Null
$Failures = [Collections.Generic.List[object]]::new()
$FailureLog = Join-Path $PSScriptRoot ('DOWNLOAD_FAILURES_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N') + '.csv')
$Skipped = 0
$Downloaded = 0
Write-Host ("Package: {0} files, {1:N4} GiB. Server files will remain unchanged." -f $Items.Count, ($TotalBytes / 1GB))

foreach ($Item in $Items) {
    $Row = $Item.Row
    $Target = $Item.Target
    $Partial = $Target + '.download.partial'
    $Reason = 'LOCAL_IO_OR_VALIDATION_FAILED'
    try {
        Assert-SafeLocalPath $Target
        Assert-SafeLocalPath $Partial
        if (Test-Path -LiteralPath $Target) {
            if (Test-VerifiedFile $Target $Row) {
                $Skipped++
                Write-Host ("SKIP verified: {0}" -f $Row.windows_relative_path)
                continue
            }
            $Reason = 'EXISTING_FINAL_FILE_MISMATCH_LEFT_UNCHANGED'
            throw 'Existing final file differs; manual review is required.'
        }
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Target)) | Out-Null
        if ((Test-Path -LiteralPath $Partial) -and -not (Test-Path -LiteralPath $Partial -PathType Leaf)) {
            throw 'Partial path is not a regular file.'
        }
        $Success = $false
        for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
            Write-Host ("GET [{0}/{1}]: {2}" -f $Attempt, $MaxAttempts, $Row.windows_relative_path)
            # 只读取远端单个文件；复用现有 SSH alias，不绕过主机密钥校验。
            $ScpArgs = @('-q', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                "${SshAlias}:$($Row.server_absolute_path)", $Partial)
            & $Scp @ScpArgs
            $TransferCode = $LASTEXITCODE
            if ($TransferCode -eq 0 -and (Test-VerifiedFile $Partial $Row)) {
                if (Test-Path -LiteralPath $Target) { throw 'Final target appeared during transfer.' }
                Move-Item -LiteralPath $Partial -Destination $Target
                $Success = $true
                $Downloaded++
                Write-Host ("OK verified: {0}" -f $Row.windows_relative_path)
                break
            }
            Write-Warning ("Transfer or verification failed: {0}" -f $Row.windows_relative_path)
            if ($Attempt -lt $MaxAttempts) { Start-Sleep -Seconds 2 }
        }
        if (-not $Success) {
            $Reason = 'TRANSFER_OR_CHECKSUM_FAILED_AFTER_RETRIES'
            throw 'Retries exhausted; partial file retained locally.'
        }
    } catch {
        # 逐次写入失败清单，允许继续处理其他文件；不删除或覆盖已有正式本地文件。
        $Failures.Add([pscustomobject]@{
            windows_relative_path = $Row.windows_relative_path
            server_absolute_path = $Row.server_absolute_path
            reason = $Reason
        })
        $Failures | Export-Csv -LiteralPath $FailureLog -NoTypeInformation -Encoding UTF8
        Write-Warning ("FAILED: {0} ({1})" -f $Row.windows_relative_path, $Reason)
    }
}

Write-Host ("Downloaded: {0}; verified/skipped: {1}; failed: {2}." -f $Downloaded, $Skipped, $Failures.Count)
if ($Failures.Count -gt 0) {
    Write-Host ("INCOMPLETE. Review {0}; rerun the same script after resolving failures." -f $FailureLog)
    exit 1
}
Write-Host ("COMPLETE: {0}" -f $Root)
exit 0
