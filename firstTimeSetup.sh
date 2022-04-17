#!/bin/bash
set -xeuf -o pipefail

poetry install
pre-commit install
