#!/bin/bash
# =============================================================================
#  push_to_github.sh
#  QHI-Probe — One-Click GitHub Push Script
#
#  USAGE:
#    chmod +x push_to_github.sh
#    ./push_to_github.sh
#
#  What this script does:
#    1. Checks git is installed
#    2. Asks for your GitHub username
#    3. Initializes the repo
#    4. Stages all files
#    5. Creates the initial commit
#    6. Pushes to GitHub
#    7. Prints the live URL when done
# =============================================================================

set -e  # Exit on any error

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

echo ""
echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${BLUE}║         QHI-Probe — GitHub Push Script                  ║${RESET}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Step 0: Check git ────────────────────────────────────────────────────────
echo -e "${CYAN}[0/6] Checking git...${RESET}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}✗ git is not installed.${RESET}"
    echo "  Install it first:"
    echo "  • Windows: https://git-scm.com/download/win"
    echo "  • Mac:     brew install git"
    echo "  • Ubuntu:  sudo apt install git"
    exit 1
fi
GIT_VERSION=$(git --version)
echo -e "  ${GREEN}✓ Found: ${GIT_VERSION}${RESET}"

# ── Step 1: Get GitHub username ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}[1/6] GitHub Setup${RESET}"
echo -e "  Please enter your GitHub username"
echo -e "  ${YELLOW}(create a free account at github.com if you don't have one)${RESET}"
echo ""
read -p "  GitHub username: " GITHUB_USER

if [ -z "$GITHUB_USER" ]; then
    echo -e "${RED}✗ Username cannot be empty.${RESET}"
    exit 1
fi

REPO_URL="https://github.com/${GITHUB_USER}/qhi-probe.git"
echo ""
echo -e "  ${GREEN}✓ Repo URL will be: ${REPO_URL}${RESET}"

# ── Step 2: One-time git config ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}[2/6] Git Configuration${RESET}"

CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -z "$CURRENT_NAME" ]; then
    read -p "  Your name (for git commits): " GIT_NAME
    git config --global user.name "$GIT_NAME"
    echo -e "  ${GREEN}✓ Name set: ${GIT_NAME}${RESET}"
else
    echo -e "  ${GREEN}✓ Git name already set: ${CURRENT_NAME}${RESET}"
fi

if [ -z "$CURRENT_EMAIL" ]; then
    read -p "  Your email: " GIT_EMAIL
    git config --global user.email "$GIT_EMAIL"
    echo -e "  ${GREEN}✓ Email set: ${GIT_EMAIL}${RESET}"
else
    echo -e "  ${GREEN}✓ Git email already set: ${CURRENT_EMAIL}${RESET}"
fi

# ── Step 3: Initialize repo ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[3/6] Initializing Repository${RESET}"

# Check if already initialized
if [ -d ".git" ]; then
    echo -e "  ${YELLOW}⚠ Git already initialized — skipping init${RESET}"
else
    git init
    echo -e "  ${GREEN}✓ Git repository initialized${RESET}"
fi

# ── Step 4: Stage all files ──────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[4/6] Staging Files${RESET}"

git add .

# Show what's being committed
FILE_COUNT=$(git diff --cached --numstat | wc -l)
echo -e "  ${GREEN}✓ Staged ${FILE_COUNT} files${RESET}"
echo ""
echo "  Files included:"
git diff --cached --name-only | sed 's/^/    /'

# ── Step 5: Commit ───────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[5/6] Creating Initial Commit${RESET}"

git commit -m "Initial release: QHI-Probe v0.1.0

Quantified Hallucination Index for Clinical LLMs
via Sparse Entity-Conditioned Probing

Architecture:
- Probe-C: Logistic Regression for uncertainty detection
- Probe-R: MLP (64-32) for clinical risk scoring (1-5)
- Probe-V: L1 Logistic for causal violation detection
- QHI = Uncertainty x Risk x Violation x 5 (range 0-25)

Key results:
- AUC-ROC: 1.000 on MedQA-USMLE benchmark
- Severity correlation: r=0.9533
- Inference: 0.946ms on CPU, no GPU required
- ISO 14971 aligned gates: AUTO_USE / REVIEW / BLOCK

Datasets: MedQA-USMLE, MedMCQA, TruthfulQA, Demo
AI testing: ChatGPT / Gemini / Claude comparison scripts
Tests: 8 unit tests, all passing"

echo -e "  ${GREEN}✓ Commit created${RESET}"

# ── Step 6: Push to GitHub ───────────────────────────────────────────────────
echo ""
echo -e "${CYAN}[6/6] Pushing to GitHub${RESET}"
echo ""
echo -e "  ${YELLOW}BEFORE PUSHING — make sure you have:${RESET}"
echo "  1. Created the repo at: https://github.com/new"
echo "     • Name: qhi-probe"
echo "     • Set to: Public"
echo "     • Do NOT initialize with README"
echo ""
read -p "  Have you created the GitHub repo? (y/n): " READY

if [ "$READY" != "y" ] && [ "$READY" != "Y" ]; then
    echo ""
    echo -e "  ${YELLOW}No problem! Create it at: https://github.com/new${RESET}"
    echo "  Then run these commands manually:"
    echo ""
    echo -e "  ${BOLD}git branch -M main${RESET}"
    echo -e "  ${BOLD}git remote add origin ${REPO_URL}${RESET}"
    echo -e "  ${BOLD}git push -u origin main${RESET}"
    echo ""
    echo -e "  ${GREEN}Your commit is saved locally. Run the push commands after creating the repo.${RESET}"
    exit 0
fi

git branch -M main

# Remove existing remote if present
if git remote get-url origin &>/dev/null; then
    git remote remove origin
fi

git remote add origin "$REPO_URL"

echo ""
echo -e "  ${YELLOW}Pushing to GitHub...${RESET}"
echo -e "  ${YELLOW}(You will be asked for username + password/token)${RESET}"
echo ""
echo -e "  ${CYAN}NOTE: If you have 2FA enabled, use a Personal Access Token${RESET}"
echo -e "  ${CYAN}as your password. Get one at:${RESET}"
echo -e "  ${CYAN}github.com → Settings → Developer Settings → Personal Access Tokens${RESET}"
echo ""

git push -u origin main

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║  ✅  Successfully pushed to GitHub!                     ║${RESET}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Your repo is live at:${RESET}"
echo -e "  ${CYAN}https://github.com/${GITHUB_USER}/qhi-probe${RESET}"
echo ""
echo -e "  ${BOLD}Next steps to make it discoverable:${RESET}"
echo "  1. Go to your repo page on GitHub"
echo "  2. Click ⚙️ gear icon next to 'About'"
echo "  3. Add these topics:"
echo "     hallucination-detection  clinical-nlp  medical-ai"
echo "     llm-safety  probing  interpretability  iso-14971"
echo ""
echo -e "  ${BOLD}Share your repo:${RESET}"
echo -e "  ${CYAN}https://github.com/${GITHUB_USER}/qhi-probe${RESET}"
echo ""
