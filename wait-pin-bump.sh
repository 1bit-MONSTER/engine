#!/usr/bin/env bash
# Wait for the engine main pin to move off 1d32c22609d6 (the #786 bump) to the new fork tip, so #788's closing
# evidence can be taken on an engine that actually contains #172 + #173. Prints the new pin when it lands.
set -u
cd ~/wt/i680-engine
old=1d32c22609d6eb23d48a26ba14e680e40afd6e17
for i in $(seq 180); do
    git fetch -q origin main 2>/dev/null
    pin=$(git ls-tree origin/main third_party/llama.cpp 2>/dev/null | awk '{print $3}')
    if [ -n "${pin:-}" ] && [ "$pin" != "$old" ]; then
        echo "PIN MOVED at $(date '+%F %T'): third_party/llama.cpp -> $pin"
        git log --oneline -1 origin/main
        exit 0
    fi
    sleep 60
done
echo "PIN DID NOT MOVE within 3 h (still $old) at $(date '+%F %T')"
exit 1
