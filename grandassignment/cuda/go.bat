@echo off
nvcc -Xcompiler "/wd 4819"  %1 -O3 -arch=sm_20
if %errorlevel% == 1 (
    exit
)
.\a.exe < ./input/input.txt > ./input/output.txt