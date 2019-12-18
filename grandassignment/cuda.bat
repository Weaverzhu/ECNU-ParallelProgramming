@echo off
nvcc -Xcompiler "/wd 4819"  %1 -O3 -o cuda
if %errorlevel% == 1 (
    exit
)
.\cuda.exe