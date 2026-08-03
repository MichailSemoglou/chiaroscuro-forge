# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in Chiaroscuro Forge, please report it
privately rather than opening a public issue. Send details to
**[m.semoglou@tongji.edu.cn](mailto:m.semoglou@tongji.edu.cn)**.

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce
- The affected version(s)
- Any suggested mitigations

You will receive an acknowledgment within 48 hours and a timeline for a fix
shortly after. We aim to release patches for confirmed vulnerabilities within
7 days.

## Scope

The security policy covers:

- The Python package distributed on PyPI
- The REST API endpoints exposed by `chiaroscuro_forge.api`
- Image processing inputs that could trigger memory or filesystem attacks
- Authentication and rate-limiting mechanisms

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Deployment notes

- The REST API stores API keys and job state in memory. This is appropriate for
  a single-process deployment, such as one app instance or one container replica,
  where restarts are expected to reset those values.
- For consistent API keys and job state across restarts or multiple workers, use
  a shared durable backend such as Redis or a database and wire the service to
  that backend. The current implementation does not provide this persistence layer.
- Set the `CHIAROSCURO_API_KEY` environment variable to pre-provision an
  initial API key before exposing the server on a network.
- Run the server behind TLS (reverse proxy with Let's Encrypt, or use the
  `ssl_keyfile`/`ssl_certfile` parameters in `run_server`).
