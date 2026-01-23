$projectRoot = "d:\Documents\Bus Project\Mini Bus"
$uploadDir = Join-Path $projectRoot "autodl_upload"
$projectRootFull = [System.IO.Path]::GetFullPath($projectRoot)
$uploadDirFull = [System.IO.Path]::GetFullPath($uploadDir)

$syncPaths = @(
    @{Source="per"; Include=@("*.py", "*.md", "LICENSE"); Exclude=@("__pycache__")},
    @{Source="src"; Include=@("*.py"); Exclude=@("__pycache__")},
    @{Source="scripts"; Include=@("*.py"); Exclude=@("__pycache__")},
    @{Source="configs"; Include=@("*.yaml"); Exclude=@()},
    @{Source="tests"; Include=@("*.py"); Exclude=@("__pycache__")},
    @{Source="docs"; Include=@("*.md"); Exclude=@()},
    @{Source="."; Include=@("requirements.txt", "pytest.ini"); Exclude=@()}
)

$updatedFiles = @()
$newFiles = @()

Write-Host "Starting sync to autodl_upload..." -ForegroundColor Cyan

foreach ($pathConfig in $syncPaths) {
    $sourcePath = Join-Path $projectRoot $pathConfig.Source
    $sourcePathFull = [System.IO.Path]::GetFullPath($sourcePath)
    
    if (-not (Test-Path $sourcePath)) {
        continue
    }
    
    $sourceFiles = Get-ChildItem -Path $sourcePath -Recurse -File | Where-Object {
        $file = $_
        if ($file.FullName.StartsWith($uploadDirFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $false
        }
        $relativePath = $file.FullName.Substring($sourcePathFull.Length + 1)
        
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
        $relativePath = $sourceFile.FullName.Substring($projectRootFull.Length + 1)
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

# Write BUILD_ID.txt into upload bundle
$buildId = & python -c "from src.utils.build_info import get_build_id; print(get_build_id())"
$buildIdPath = Join-Path $uploadDir "BUILD_ID.txt"
Set-Content -Path $buildIdPath -Value $buildId -Encoding ascii
Write-Host "BUILD_ID: $buildId" -ForegroundColor Gray

$zipPath = Join-Path $projectRoot "autodl_upload.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

$zipItems = Get-ChildItem -Path $uploadDir -Force | Where-Object { $_.Name -ne "autodl_upload" }
Compress-Archive -Path $zipItems.FullName -DestinationPath $zipPath -Force
Write-Host "Created: autodl_upload.zip" -ForegroundColor Green

$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host "Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Gray
