# Security Policy

## Supported Versions

This project is in early development. Security fixes are applied to the latest
release on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not open a public
issue.** Instead, report it privately:

- Email: **kadiryonak13@gmail.com**
- Or use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We will acknowledge your report within a few days and keep you informed of the
progress toward a fix.

## API Keys & Secrets

This project uses third-party LLM and search APIs. **Never commit your `.env`
file or API keys.** The `.gitignore` already excludes `.env`. If you
accidentally commit a key, rotate it immediately.
