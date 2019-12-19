@echo off
nvcc -Xcompiler "/wd 4819"  %1 -O3 -o cuda -arch=sm_60
if %errorlevel% == 1 (
    exit
)
.\cuda.exe