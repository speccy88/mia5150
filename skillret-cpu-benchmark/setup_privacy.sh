#!/usr/bin/env sh
# PROSPECTIVE PRIVACY SETUP, added after measurements; not the measured config.
# Source this file before launching a new Python process that may initialize ORT:
#   . ./setup_privacy.sh
# This file only sets an environment variable. It does not launch any process.
# No inference was executed to validate this newly provided setup.
export ORT_DISABLE_TELEMETRY=1
