#!/usr/bin/env bash
set -euo pipefail

config_dir="$HOME/.config/openai"
env_file="$config_dir/openai.env"

if [[ -e "$env_file" ]]; then
    printf 'Key file already exists at %s; refusing to overwrite it.\n' "$env_file" >&2
    exit 1
fi

read -r -s -p 'OpenAI API key (input hidden): ' api_key
printf '\n'
if [[ -z "$api_key" ]]; then
    printf 'No key entered; nothing was written.\n' >&2
    exit 1
fi

umask 077
mkdir -p "$config_dir"
temporary_file="$(mktemp "$config_dir/openai.env.XXXXXX")"
trap 'rm -f "$temporary_file"' EXIT
chmod 600 "$temporary_file"
printf 'export OPENAI_API_KEY=%q\n' "$api_key" > "$temporary_file"
mv "$temporary_file" "$env_file"
trap - EXIT
unset api_key
printf 'Saved a private OpenAI environment file at %s (mode 600).\n' "$env_file"