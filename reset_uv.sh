#!/bin/sh
# need uv installed already

# remove previous install
toml unset --toml-path pyproject.toml project.dependencies
rm -rf ./.venv/

# create new install
uv venv --python 3.14.6 --clear
uv add -r requirements.in
uv tool install toml-cli
uv tool install black
uv tool install blacken-docs
uv tool install blackdoc
uv tool install pre-commit
