$ErrorActionPreference = "Stop"
$Python = "C:\Users\EDY\.conda\envs\py310\python.exe"
& $Python -B "$PSScriptRoot\train.py"
