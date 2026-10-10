# Repository workflow

- Keep all repository code and documentation on `main`.
- Do not create or push another branch unless the user explicitly requests it.
- Update from remote `main` before making changes, and publish commits directly to `main`.
- Before removing a temporary branch, verify that every commit it contains is reachable from `main`.

## Public repository privacy

- Use the generic commit author `Research Team` with `user.email=anonymous`.
- Keep team names, personal email addresses, student identifiers, personal home paths, credentials and passwords out of repository files, notebook outputs, PDFs, archives, and commit messages.
- Use a fresh clone of the public repository. Never merge or force-push history from an older private clone.
- Run `gitleaks git --config .gitleaks.toml --redact=100 --log-opts=--all` before publication. Reviewed exceptions cover benchmark hashes and public identifiers only; do not add exceptions for credentials.
- Load credentials from local environment variables. Keep `.env`, key files and private backups outside this repository.
- Privacy redaction changes archived text and file hashes. Preserve measured metrics and model parameters; describe redactions and refresh current publication checksums.

- Team credit may use only the permitted aliases `deepujoy88`, `speccy88`, and `vsimard`; never add their real names, emails or student identifiers.
