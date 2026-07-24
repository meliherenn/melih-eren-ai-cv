# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected secret leak or exploitable vulnerability. Contact
[meliheren2834@gmail.com](mailto:meliheren2834@gmail.com) with a concise description, affected revision,
reproduction steps and potential impact.

## Secrets and deployment

- Never commit API keys, tokens, passwords, `.env` files or `.streamlit/secrets.toml`.
- Configure provider credentials and the optional admin password through deployment secrets.
- The public admin panel is disabled unless both `ENABLE_ADMIN_PANEL=true` and `ADMIN_PASSWORD` are set.
- Rotate any credential that may have appeared in logs, screenshots, commits or shared configuration.
- The application verifies checksums for the repository-owned FAISS index and JSON metadata before loading them.
- Retrieval metadata uses JSON; the application does not load Python pickle files.

## Supported version

Security fixes are applied to the latest commit on the `main` branch.
