#!/usr/bin/env bash
# Wrapper: the device half under ONE npu-hold ticket, 9 minutes max (plan section 2).
exec "$HOME/.local/bin/npu-hold" -t 9 -l "w3b-olmo-2048x6144" -- "$HOME/ogaload/oga13/w3b-device-inner.sh"
