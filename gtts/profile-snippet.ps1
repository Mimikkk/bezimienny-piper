function gtts {
    if (-not $args -or $args.Count -eq 0) {
        Write-Host 'Uzycie: gtts "tekst"'
        return
    }
    & 'C:\Program Files (x86)\Steam\steamapps\common\Gothic II\system\python_runtime_piper\python.exe' 'C:\Users\mimi\Documents\gtts\gtts.py' @args
}
