#!/bin/bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/../.git/hooks"

# Create symlinks for all hooks from tools/hook into the .git/hooks older 
ln -fs ../../tools/hooks/* .