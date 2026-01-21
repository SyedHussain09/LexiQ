<#
.SYNOPSIS
  Development helper script for the UAE Legal RAG project.
.DESCRIPTION
  Provides Bootstrap, Run, Test, and Format actions.
.PARAMETER Bootstrap
  Creates venv, installs dependencies, copies .env.example to .env.
.PARAMETER Run
  Starts the Streamlit app.
.PARAMETER Test
  Runs pytest.
.PARAMETER Format
  Runs ruff linter and formatter.
.PARAMETER Python
  Python version to use (default: 3.13).
#>
param(
  [switch]$Bootstrap,
  [switch]$Run,
  [switch]$Test,
  [switch]$Format,
  [string]$Python = "3.13"
)

$ErrorActionPreference = 'Stop'

function New-VirtualEnvironment {
  <#
  .SYNOPSIS
    Creates Python virtual environment if it doesn't exist.
  #>
  if (-not (Test-Path .venv)) {
    Write-Host "Creating venv with Python $Python..." -ForegroundColor Cyan
    py -$Python -m venv .venv
  }
}

function Enter-VirtualEnvironment {
  <#
  .SYNOPSIS
    Activates the Python virtual environment.
  #>
  . .\.venv\Scripts\Activate.ps1
}

function Install-Dependencies {
  <#
  .SYNOPSIS
    Installs Python dependencies and the package in editable mode.
  #>
  Write-Host "Installing dependencies..." -ForegroundColor Cyan
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  Write-Host "Installing package in editable mode..." -ForegroundColor Cyan
  python -m pip install -e .
}

# Bootstrap: create venv, install deps, copy .env
if ($Bootstrap) {
  New-VirtualEnvironment
  Enter-VirtualEnvironment
  Install-Dependencies
  if (-not (Test-Path .env)) {
    Copy-Item .env.example .env -ErrorAction SilentlyContinue
  }
  Write-Host "`nBootstrap complete! Edit .env and set OPENAI_API_KEY." -ForegroundColor Green
  exit 0
}

# Ensure venv is active for all other commands
New-VirtualEnvironment
Enter-VirtualEnvironment

if ($Test) {
  Write-Host "Running tests..." -ForegroundColor Cyan
  python -m pytest -q
  exit $LASTEXITCODE
}

if ($Format) {
  Write-Host "Running linter and formatter..." -ForegroundColor Cyan
  python -m ruff check . --fix
  python -m ruff format .
  Write-Host "Formatting complete." -ForegroundColor Green
  exit 0
}

if ($Run) {
  Write-Host "Starting Streamlit app..." -ForegroundColor Cyan
  .\.venv\Scripts\streamlit.exe run app.py
  exit 0
}

Write-Host @"

UAE Legal RAG - Development Script
===================================
Usage:
  .\scripts\dev.ps1 -Bootstrap   Create venv, install deps, copy .env
  .\scripts\dev.ps1 -Run         Start Streamlit app
  .\scripts\dev.ps1 -Test        Run pytest
  .\scripts\dev.ps1 -Format      Run ruff linter + formatter

Options:
  -Python <version>              Python version (default: 3.13)

"@ -ForegroundColor Yellow
