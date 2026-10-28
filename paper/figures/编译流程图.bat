@echo off
chcp 65001 >nul
setlocal
set PATH=C:\texlive\2026\bin\windows;%PATH%
cd /d "%~dp0"

set TEX=%~1
if "%TEX%"=="" set TEX=q1_methodology_flowchart.tex

echo ============================================
echo   Compiling %TEX%  (XeLaTeX)
echo ============================================

rem --- 第一遍：生成 .aux ---
xelatex -interaction=nonstopmode -halt-on-error "%TEX%"
if errorlevel 1 (
    echo.
    echo [FAIL] 编译出错，请看上面的 "! " 开头的错误行。
    echo        常见原因：文件名打错 / 不是本目录下的 .tex
    pause
    exit /b 1
)

rem --- 第二遍：解析交叉引用 ---
xelatex -interaction=nonstopmode -halt-on-error "%TEX%"
if errorlevel 1 (
    echo.
    echo [FAIL] 第二遍出错。
    pause
    exit /b 1
)

echo.
echo [OK] 完成：%TEX:.tex=.pdf%
pause
