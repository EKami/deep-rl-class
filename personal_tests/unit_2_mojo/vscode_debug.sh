#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

magic run mojo debug --vscode "$SCRIPT_DIR/src/q_learning_mojo/main.mojo"
