#!/usr/bin/env bash
# Render build command: ./build.sh
set -o errexit

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt