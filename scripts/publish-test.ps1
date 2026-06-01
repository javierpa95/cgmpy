# Upload built distributions to Test PyPI (Windows / PowerShell).
#
# Usage:
#   pwsh -File scripts/publish-test.ps1
#
# Requires: env var TEST_PYPI_TOKEN (or TWINE_USERNAME + TWINE_PASSWORD).

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

# Credentials
if (-not $env:TEST_PYPI_TOKEN -and -not $env:TWINE_USERNAME) {
  Write-Error "Set TEST_PYPI_TOKEN (or TWINE_USERNAME+TWINE_PASSWORD) in the environment."
  Write-Error "For CI, prefer Trusted Publishing (OIDC); see .github\workflows\publish-pypi.yml."
  exit 1
}

if ($env:TEST_PYPI_TOKEN) {
  $env:TWINE_USERNAME = '__token__'
  $env:TWINE_PASSWORD = $env:TEST_PYPI_TOKEN
}

$repo = if ($env:TWINE_REPOSITORY) { $env:TWINE_REPOSITORY } else { 'testpypi' }

Write-Host "==> Uploading to Test PyPI (repository: $repo)"
& python -m twine upload --repository $repo (Join-Path $distPath '*')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Done. Check https://test.pypi.org/project/cgmpy/ to verify."
Write-Host "    Install with: pip install --index-url https://test.pypi.org/simple/ cgmpy"
