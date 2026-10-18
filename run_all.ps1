$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if (Get-Command py -ErrorAction SilentlyContinue) {
    # Windows PowerShell 5 会把原生命令的 stderr 提升为终止错误，需捕获后回退。
    $lab2_launcher_ok = $false
    try {
        & py -3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>$null
        $lab2_launcher_ok = ($LASTEXITCODE -eq 0)
    } catch {
        $lab2_launcher_ok = $false
    }
    if ($lab2_launcher_ok) {
        & py -3 run.py @args
        exit $LASTEXITCODE
    }
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ is required.' }
    & python run.py @args
} else {
    throw 'Python 3.10+ was not found. Install Python and enable its launcher or PATH.'
}
exit $LASTEXITCODE
