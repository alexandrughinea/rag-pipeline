#!/bin/bash

set -e  # Exit on error
echo "🔍 Formatting Python code..."
ruff check --fix .