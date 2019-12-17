@echo off
g++ %1 -o main.exe -std=c++0x -O2 -Wall
.\main.exe < ../input/input_10.txt