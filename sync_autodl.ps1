$projectRoot = "d:\Documents\Bus Project\Mini Bus"
$uploadDir = Join-Path $projectRoot "autodl_upload"

$syncPaths = @(
    @{Source="src"; Include=@("*.py"); Exclude=@("__pycache__")},
    @{Source="scripts"; Include=@("*.py"); Exclude=@()},
    @{Source="configs"; Include=@("*.yaml"); Exclude=@()},
    @{Source="tests"; Include=@("*.py"); Exclude=@("__pycache__")},
    @{Source="."; Include=@("requirements.txt", "pytest.ini"); Exclude=@()}
)

$updatedFiles = @()
$newFiles = @()

Write-Host "Starting sync to autodl_upload..." -ForegroundColor Cyan

foreach ($pathConfig in $syncPaths) {
    $sourcePath = Join-Path $projectRoot $pathConfig.Source
    
    if (-not (Test-Path $sourcePath)) {
        continue
    }
    
    $sourceFiles = Get-ChildItem -Path $sourcePath -Recurse -File | Where-Object {
        $file = $_
        $relativePath = $file.FullName.Substring($sourcePath.Length + 1)
        
        $matchInclude = $false
        foreach ($pattern in $pathConfig.Include) {
            if ($file.Name -like $pattern) {
                $matchInclude = $true
                break
            }
        }
        
        $excluded = $false
        foreach ($pattern in $pathConfig.Exclude) {
            if ($relativePath -like "*$pattern*") {
                $excluded = $true
                break
            }
        }
        
        $matchInclude -and (-not $excluded)
    }
    
    foreach ($sourceFile in $sourceFiles) {
        $relativePath = $sourceFile.FullName.Substring($projectRoot.Length + 1)
        $targetFile = Join-Path $uploadDir $relativePath
        
        $needCopy = $false
        $isNew = $false
        
        if (Test-Path $targetFile) {
            $sourceTime = $sourceFile.LastWriteTime
            $targetTime = (Get-Item $targetFile).LastWriteTime
            
            if ($sourceTime -gt $targetTime) {
                $needCopy = $true
                Write-Host "UPDATE: $relativePath" -ForegroundColor Green
            }
        } else {
            $needCopy = $true
            $isNew = $true
            Write-Host "NEW: $relativePath" -ForegroundColor Cyan
        }
        
        if ($needCopy) {
            $targetDir = Split-Path $targetFile -Parent
            if (-not (Test-Path $targetDir)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            
            Copy-Item -Path $sourceFile.FullName -Destination $targetFile -Force
            
            if ($isNew) {
                $newFiles += $relativePath
            } else {
                $updatedFiles += $relativePath
            }
        }
    }
}

Write-Host "`nSync complete!" -ForegroundColor Green
Write-Host "New files: $($newFiles.Count)" -ForegroundColor Cyan
Write-Host "Updated files: $($updatedFiles.Count)" -ForegroundColor Yellow

$zipPath = Join-Path $projectRoot "autodl_upload.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path $uploadDir -DestinationPath $zipPath -Force
Write-Host "Created: autodl_upload.zip" -ForegroundColor Green

$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host "Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Gray
