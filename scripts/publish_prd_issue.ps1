# Publish project PRD to GitHub Issues with ready-for-agent label.
# Requires: gh auth login
param(
    [string]$PrdPath = "$PSScriptRoot\..\artefacts\prd\face-ai-product.md",
    [string]$Title = "PRD: face-ai — Face Shape & Seasonal Color Analysis Platform"
)

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Error "GitHub CLI (gh) not found. Install: winget install GitHub.cli"
    exit 1
}

$auth = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Not logged in. Run: gh auth login"
    exit 1
}

if (-not (Test-Path $PrdPath)) {
    Write-Error "PRD file not found: $PrdPath"
    exit 1
}

$body = Get-Content -Raw -Encoding UTF8 $PrdPath
$issueUrl = gh issue create --repo NXTHlNG/face-ai --title $Title --body $body --label "ready-for-agent" 2>&1
if ($LASTEXITCODE -ne 0) {
    $issueUrl = gh issue create --repo NXTHlNG/face-ai --title $Title --body $body
    if ($issueUrl -match '#(\d+)') {
        gh label create "ready-for-agent" --description "PRD ready for agent implementation" --color "0E8A16" --repo NXTHlNG/face-ai 2>$null
        gh issue edit $Matches[1] --repo NXTHlNG/face-ai --add-label "ready-for-agent"
    }
}

Write-Output $issueUrl
