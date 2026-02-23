@echo off
REM =============================================================================
REM  push_to_github.bat
REM  QHI-Probe — GitHub Push Script for Windows
REM
REM  USAGE: Double-click this file, or run in Command Prompt
REM =============================================================================

echo.
echo ==============================================================
echo    QHI-Probe - GitHub Push Script (Windows)
echo ==============================================================
echo.

REM Check git
git --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo [OK] git found

REM Get GitHub username
echo.
set /p GITHUB_USER="Enter your GitHub username: "

if "%GITHUB_USER%"=="" (
    echo ERROR: Username cannot be empty.
    pause
    exit /b 1
)

set REPO_URL=https://github.com/%GITHUB_USER%/qhi-probe.git
echo.
echo Repo URL: %REPO_URL%

REM Git config
echo.
echo [1/5] Checking git configuration...
git config --global user.name >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set /p GIT_NAME="Your name for commits: "
    git config --global user.name "%GIT_NAME%"
)
git config --global user.email >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    set /p GIT_EMAIL="Your email for commits: "
    git config --global user.email "%GIT_EMAIL%"
)
echo [OK] Git configured

REM Initialize
echo.
echo [2/5] Initializing repository...
if not exist ".git" (
    git init
)
echo [OK] Repository initialized

REM Stage files
echo.
echo [3/5] Staging all files...
git add .
echo [OK] Files staged

REM Commit
echo.
echo [4/5] Creating commit...
git commit -m "Initial release: QHI-Probe v0.1.0 — Clinical LLM hallucination scoring"
echo [OK] Commit created

REM Instructions before push
echo.
echo ==============================================================
echo  BEFORE PUSHING - Make sure you have created the repo:
echo.
echo  1. Go to: https://github.com/new
echo  2. Name: qhi-probe
echo  3. Set to: Public
echo  4. Do NOT initialize with README
echo  5. Click Create repository
echo ==============================================================
echo.
pause

REM Push
echo.
echo [5/5] Pushing to GitHub...
git branch -M main
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git push -u origin main

echo.
echo ==============================================================
echo  SUCCESS! Your repo is live at:
echo  https://github.com/%GITHUB_USER%/qhi-probe
echo ==============================================================
echo.
echo Next: Add topics on GitHub for discoverability:
echo   hallucination-detection  clinical-nlp  medical-ai
echo   llm-safety  probing  iso-14971
echo.
pause
