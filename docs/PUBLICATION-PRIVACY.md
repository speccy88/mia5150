# Public publication and privacy

The public repository starts with a new anonymous snapshot on `main`. Earlier private commits were not uploaded to this repository. The original history is preserved separately in a private archive and private local backup. This prevents the public repository from exposing personal details through earlier commit URLs.

## What was sanitized

Real team names, personal email addresses, student identifiers and personal home paths were removed from project content. The proposal PDF was redacted in its content streams, rather than covered with an overlay. PDF author metadata and notebook execution outputs were removed. The handoff ZIP was unpacked, sanitized and rebuilt. Credential-shaped examples in stored dataset and request text were replaced with redaction placeholders, including the public Stripe documentation test-key sample. No environment or private key files are included.

The recorded benchmark metrics and model parameters were retained. Redacted request text differs from the original recorded prompt: original request hashes, provider token counts and costs describe the original private execution. They must not be presented as measurements of the redacted text. Historical source inventories describe the original run; current publication checksums describe the sanitized files. Fetch the pinned public dataset for fresh reproduction, and save new measurements separately.

The permitted team aliases are `deepujoy88`, `speccy88`, and `vsimard`.

Upstream author names in references and required third-party license notices are retained. GitHub exposes the repository owner's account handle and public account profile independently of files or commit metadata.

## Safe updates

Use a fresh clone and configure the repository locally:

```sh
git config user.name 'Research Team'
git config user.email anonymous
git config core.hooksPath .githooks
```

Work directly on `main`. GitHub rules prevent creation of additional code branches, and `main` rejects deletion, force pushes and merge commits. Secret scanning and push protection are enabled. Never merge, mirror-push or force-push an old private checkout into this repository. Such a push could reintroduce the private history. Install Gitleaks and run:

```sh
gitleaks dir --config .gitleaks.toml --redact=100 --max-archive-depth 3 .
gitleaks git --config .gitleaks.toml --redact=100 --log-opts=--all .
```

The scan configuration retains default credential rules. Its narrow exceptions cover reviewed benchmark case hashes, JSON field names, and documented public identifiers. A credential must be removed, and an actual exposed credential must also be rotated or revoked. Never solve a credential finding by adding an exception.

Before publishing, also review document metadata, notebook output, nested archives, and commit author details for personal information. Keep any private review report and original backup outside the public checkout.
