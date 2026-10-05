@echo off
REM Save this machine's per-dataset seed splits to seed_snapshot\ (see scripts\export_seeds.py)
REM   scripts\export_seeds.bat
REM   scripts\export_seeds.bat --datasets cora pubmed --seeds 42 43 44 45
cd /d "%~dp0.."
python scripts\export_seeds.py %*
