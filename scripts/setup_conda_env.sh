#!/bin/bash


# --- Safety checks ---
if [ -z "$1" ]; then
    echo "Error: No conda environment name provided." >&2
    # In a sourced script, 'return' is used instead of 'exit'
    # to stop execution of this script but not the parent script.
    return 1
fi

# --- Configuration ---
# The base path where all your conda environments are stored.
readonly CONDA_BASE_PATH=""
readonly CONDA_ENV_NAME=$1
readonly ENV_BIN_PATH="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin"

# --- Main Logic ---
if [ ! -d "${ENV_BIN_PATH}" ]; then
    echo "Error: Conda environment '${CONDA_ENV_NAME}' not found." >&2
    echo "       Searched at: ${ENV_BIN_PATH}" >&2
    return 1
fi

# Prepend the environment's bin directory to the PATH.
export PATH="${ENV_BIN_PATH}:${PATH}"

