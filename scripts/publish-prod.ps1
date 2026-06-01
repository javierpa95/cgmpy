# Upload built distributions to PRODUCTION PyPI (Windows / PowerShell).
#
# WARNING: this is IRREVERSIBLE. Always publish to Test PyPI first.

[CmdletBinding()]
param()

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path
Set-Location -LiteralPath $RepoRoot

# Sanity
$distPath = Join-Path $RepoRoot 'dist'
if (-not (Test-Path -LiteralPath $distPath) -or -not (Get-ChildItem -LiteralPath $distPath)) {
  Write-Error "dist\ is empty. Run scripts\build-dist.ps1 first."
  exit 1
}

if (-not $env:PYPI_TOKEN -and -not $env:TWINE_USERNAME) {
  Write-Error "Set PYPI_TOKEN (or TWINE_USERNAME+TWINE_PASSWORD) in the environment."
  Write-Error "For CI, prefer Trusted Publishing (OIDC); see .github\workflows\publish-pypi.yml."
  exit 1
}

# Confirm
$version = & python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"
Write-Host "About to upload cgmpy $version to PRODUCTION PyPI."
Write-Host ""
Get-ChildItem -LiteralPath $distPath | Format-Table Name, Length -AutoSize
Write-Host ""
$ans = Read-Host "Type 'publish $version' to confirm"
if ($ans -ne "publish $version") {
  Write-Error "Aborted."
  exit 1
}

if ($env:PYPI_TOKEN) {
  $env:TWINE_USERNAME = '__token__'
  $env:TWINE_PASSWORD = $env:PYPI_TOKEN
}

$repo = if ($env:TWINE_REPOSITORY) { $env:TWINE_REPOSITORY } else { 'pypi' }

Write-Host "==> Uploading to PRODUCTION PyPI (repository: $repo)"
& python -m twine upload --repository $repo (Join-Path $distPath '*')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Done. Check https://pypi.org/project/cgmpy/ to verify."
Write-Host "    Install with: pip install cgmpy"
