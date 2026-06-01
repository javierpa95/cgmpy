# Build source and wheel distributions for CGMPy (Windows / PowerShell).
#
# Usage:
#   pwsh -File scripts/build-dist.ps1
#   pwsh -File scripts/build-dist.ps1 -NoClean
#   pwsh -File scripts/build-dist.ps1 -Sdist
#   pwsh -File scripts/build-dist.ps1 -Wheel
#
# Requires: build, twine (installed via the `dev` extra).
#
# Output: dist\cgmpy-VERSION.tar.gz and dist\cgmpy-VERSION-py3-none-any.whl

[CmdletBinding()]
param(
  [switch]$NoClean,
  [switch]$Sdist,
  [switch]$Wheel,
  [switch]$Help
)

if ($Help) {
  @"
build-dist.ps1 - Build sdist and/or wheel for CGMPy.

Usage:
  build-dist.ps1             - Clean build (sdist + wheel)
  build-dist.ps1 -NoClean    - Skip the clean step
  build-dist.ps1 -Sdist      - Build only sdist
  build-dist.ps1 -Wheel      - Build only wheel
  build-dist.ps1 -Help       - Show this help

Requires: build, twine (installed via the 'dev' extra).
"@
  exit 0
}

# Resolve repo root from script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location -LiteralPath $RepoRoot

# Sanity checks
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
  Write-Error "python is not on PATH."
  exit 1
}

$pyVer = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$ok = & python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) {
  Write-Error "Python 3.10+ is required (found $pyVer)."
  exit 1
}

$hasBuild = & python -c "import build" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "'build' is not installed. Run: pip install -e '.[dev]'"
  exit 1
}

$hasTwine = & python -c "import twine" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Error "'twine' is not installed. Run: pip install -e '.[dev]'"
  exit 1
}

# Clean
if (-not $NoClean) {
  Write-Host "==> Cleaning previous build artifacts"
  foreach ($p in @('build', 'dist')) {
    if (Test-Path -LiteralPath $p) {
      Remove-Item -LiteralPath $p -Recurse -Force
    }
  }
  Get-ChildItem -Filter '*.egg-info' -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
  }
}

# Build
Write-Host "==> Building distributions (Python $pyVer)"
$args = @()
if ($Sdist) { $args = @('--sdist') }
elseif ($Wheel) { $args = @('--wheel') }
else { $args = @('--sdist', '--wheel') }

& python -m build @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Verify
Write-Host "==> Verifying distributions with twine"
& python -m twine check (Join-Path $RepoRoot 'dist/*')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Report
Write-Host ""
Write-Host "==> Build complete. Artifacts:"
Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'dist') | Format-Table Name, Length -AutoSize
Write-Host "Next steps:"
Write-Host "  Test PyPI:    pwsh -File scripts/publish-test.ps1"
Write-Host "  Production:   pwsh -File scripts/publish-prod.ps1    (then confirm with twine upload)"
