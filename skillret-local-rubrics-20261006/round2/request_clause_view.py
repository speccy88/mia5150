"""One deterministic request-clause view. No generation, labels or tuning."""
import re
def request_view(text):
    clauses=re.split(r'(?<=[.!?;])\s+|\n+',text.strip())
    cue=re.compile(r'\b(?:need|want|please|help|must|should|implement|create|build|design|generate|configure|set up|write|add|fix|ensure)\b',re.I)
    chosen=[c.strip() for c in clauses if cue.search(c)]
    return ' '.join(chosen) if chosen else text.strip()
