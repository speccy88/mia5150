set -e
for f in /AGENTS.md /home/user /home/user/AGENTS.md /home/user/code/AGENTS.md /home/user/code/mia5150-local/AGENTS.md; do if test -f "$f"; then printf '%s\n' "$f"; cat "$f"; fi; done
find /home/user/.agents /home/user/.codex -maxdepth 3 -type f \( -name AGENTS.md -o -name MEMORY.md -o -name SKILL.md \) 2>/dev/null | head -60 || true
ls -la /home/user/code/mia5150-local
find /home/user/code/mia5150-local -maxdepth 3 -type f \( -iname '*train*' -o -iname '*family*' -o -iname '*host*' -o -iname '*partit*' -o -iname '*skillret*' \) | head -100
ls -la /home/user/code/embeddinggemma2-pilot
