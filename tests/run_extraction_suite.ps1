param(
  [string]$ApiBase = "http://127.0.0.1:8000",
  [string]$SamplesDir = ".\\tests\\sample_complaints",
  [string]$ResultsDir = ".\\tests\\results",
  [int]$TimeoutSec = 180
)

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path -Force | Out-Null }
}

Ensure-Dir $ResultsDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $ResultsDir $timestamp
Ensure-Dir $runDir

Write-Host "API Base: $ApiBase"
Write-Host "Samples:  $SamplesDir"
Write-Host "Results:  $runDir"
Write-Host ""

$files = Get-ChildItem -Path $SamplesDir -Filter "*.json" | Sort-Object Name
if (-not $files.Count) {
  Write-Host "No sample JSON files found in $SamplesDir" -ForegroundColor Yellow
  exit 1
}

$passed = 0
$failed = 0

foreach ($f in $files) {
  $name = $f.Name
  $outPath = Join-Path $runDir ($name -replace "\.json$", ".result.json")

  Write-Host "Running $name ..."
  try {
    $resp = Invoke-RestMethod `
      -Uri "$ApiBase/extract" `
      -Method Post `
      -ContentType "application/json" `
      -InFile $f.FullName `
      -TimeoutSec $TimeoutSec

    $resp | ConvertTo-Json -Depth 12 | Out-File -Encoding utf8 $outPath

    # Basic sanity checks (expand as your eval suite grows)
    if ($null -eq $resp.crime_type -or $null -eq $resp.severity -or $null -eq $resp.summary) {
      throw "Missing semantic fields (crime_type/severity/summary)"
    }
    if ($resp.transaction_id -and ($resp.transaction_id -notmatch "^\d{10,24}$")) {
      throw "Bad transaction_id format: $($resp.transaction_id)"
    }
    if ($resp.phone -and ($resp.phone -notmatch "^[6-9]\d{9}$")) {
      throw "Bad phone format: $($resp.phone)"
    }

    $passed++
    Write-Host "  OK -> $outPath" -ForegroundColor Green
  }
  catch {
    $failed++
    $errPath = Join-Path $runDir ($name -replace "\.json$", ".error.txt")
    $_ | Out-String | Out-File -Encoding utf8 $errPath
    Write-Host "  FAIL -> $errPath" -ForegroundColor Red
  }
}

Write-Host ""
Write-Host "Suite complete. Passed: $passed | Failed: $failed"
if ($failed -gt 0) { exit 1 } else { exit 0 }

