param(
    [switch]$All,
    [switch]$Data,
    [switch]$Artifacts,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$script:CleanupFailures = @()

function Remove-ProjectPath {
    param([string]$RelativePath)

    $Target = Join-Path $ProjectRoot $RelativePath
    $ResolvedParent = Resolve-Path (Split-Path $Target -Parent) -ErrorAction SilentlyContinue
    if (-not $ResolvedParent) {
        return
    }

    $ProjectRootText = [string]$ProjectRoot
    $FullTarget = [System.IO.Path]::GetFullPath($Target)
    if (-not $FullTarget.StartsWith($ProjectRootText, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside project root: $FullTarget"
    }

    if (Test-Path -LiteralPath $FullTarget) {
        if ($DryRun) {
            Write-Host "Would remove $RelativePath"
        } else {
            Write-Host "Removing $RelativePath"
            try {
                Get-ChildItem -LiteralPath $FullTarget -Recurse -Force -ErrorAction SilentlyContinue |
                    ForEach-Object { $_.Attributes = "Normal" }
                if (Test-Path -LiteralPath $FullTarget -PathType Container) {
                    (Get-Item -LiteralPath $FullTarget -Force).Attributes = "Directory"
                } else {
                    (Get-Item -LiteralPath $FullTarget -Force).Attributes = "Normal"
                }
                Remove-Item -LiteralPath $FullTarget -Recurse -Force -ErrorAction Stop
            } catch {
                Write-Warning "Could not remove ${RelativePath}: $($_.Exception.Message)"
                $script:CleanupFailures += $RelativePath
            }
        }
    }
}

function Get-RelativeProjectPath {
    param([string]$FullPath)

    $ProjectRootText = [string]$ProjectRoot
    $FullPathText = [System.IO.Path]::GetFullPath($FullPath)
    if ($FullPathText.Length -le $ProjectRootText.Length) {
        return ""
    }
    return $FullPathText.Substring($ProjectRootText.Length).TrimStart("\", "/")
}

$DefaultTargets = @(
    ".venv",
    ".pytest_cache",
    "build",
    "dist",
    "src/worldcup_predictor.egg-info"
)

$CacheTargets = @()
foreach ($SearchPath in @("scripts", "src", "tests")) {
    $FullSearchPath = Join-Path $ProjectRoot $SearchPath
    if (Test-Path -LiteralPath $FullSearchPath) {
        $CacheTargets += Get-ChildItem -Path $FullSearchPath -Recurse -Force -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -eq "__pycache__" } |
            ForEach-Object { Get-RelativeProjectPath $_.FullName }
    }
}

foreach ($Target in $DefaultTargets + $CacheTargets) {
    Remove-ProjectPath $Target
}

if ($All -or $Data) {
    Remove-ProjectPath "data/raw"
    Remove-ProjectPath "results.tsv"
    Get-ChildItem -Path $ProjectRoot -Filter "*.log" -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notlike (Join-Path $ProjectRoot ".venv*") } |
        ForEach-Object { Remove-ProjectPath (Get-RelativeProjectPath $_.FullName) }
}

if ($All -or $Artifacts) {
    $ArtifactFiles = @(
        "artifacts/autoresearch.tsv",
        "artifacts/metrics.json",
        "artifacts/model.joblib",
        "artifacts/prediction_log.tsv",
        "artifacts/training_cycles.svg",
        "artifacts/training_history.tsv",
        "artifacts/training_progress.svg"
    )
    foreach ($Target in $ArtifactFiles) {
        Remove-ProjectPath $Target
    }
}

if ($DryRun) {
    Write-Host "Dry run complete."
} else {
    if ($script:CleanupFailures.Count -gt 0) {
        Write-Warning "Cleanup completed with $($script:CleanupFailures.Count) item(s) left behind."
        Write-Warning "Close terminals/editors using these paths and rerun cleanup:"
        foreach ($Failure in $script:CleanupFailures) {
            Write-Warning "  $Failure"
        }
    } else {
        Write-Host "Cleanup complete."
    }
}
