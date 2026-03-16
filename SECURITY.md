# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them through [GitHub Security Advisories](https://github.com/abhilashjaiswal0110/unsloth-ai/security/advisories/new).

### What to Include

- Type of issue (e.g., buffer overflow, code injection, credential exposure, etc.)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

| Severity | Initial Response | Patch Target |
|----------|-----------------|--------------|
| Critical | 24 hours        | 7 days       |
| High     | 3 days          | 14 days      |
| Medium   | 7 days          | 30 days      |
| Low      | Best effort     | Next release |

## Security Measures

This repository implements the following security measures:

- **Dependency Scanning**: Dependabot monitors for vulnerable dependencies
- **Secret Scanning**: GitHub secret scanning prevents accidental credential commits
- **Code Scanning**: CodeQL analysis for static security analysis
- **Pre-commit Hooks**: Ruff linter and formatter for code quality enforcement
- **CODEOWNERS**: All changes require review by designated code owners
- **Branch Protection**: Main branch requires PR reviews before merging

## Best Practices for Contributors

1. **Never commit secrets** — API keys, tokens, passwords, or credentials
2. **Use `.env` files** for local configuration (already in `.gitignore`)
3. **Pin dependencies** to specific versions in `pyproject.toml`
4. **Review dependencies** before adding new packages
5. **Follow least privilege** — request only necessary permissions
6. **Validate inputs** — sanitize all user-provided data in training pipelines
7. **Use secure model sources** — verify model checksums from HuggingFace Hub

## Disclosure Policy

We follow responsible disclosure. We will:

1. Confirm receipt of your vulnerability report
2. Investigate and validate the reported issue
3. Develop and test a fix
4. Release the fix and publicly disclose the vulnerability

Thank you for helping keep Unsloth and its users safe.
