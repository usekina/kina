# Security and Privacy Policy

KinaBot processes sensitive voice and language data. Coordinated disclosure
helps protect participants, researchers, deployers, and the wider community.

## Supported Versions

KinaBot is an early-stage project. Security fixes are made for the current
`main` branch and the production version identified in the latest release or
deployment record. Historical prototypes and older releases may not receive
fixes. If you operate a deployment, keep your version and dependencies current.

| Version | Supported |
| --- | --- |
| Current `main` / latest release | Yes |
| Older releases and historical prototypes | No |

This table describes maintenance coverage, not a guarantee that a version is
free of vulnerabilities or suitable for clinical, safety-critical, or
high-risk use.

## Report Privately

Do **not** open a public issue for an unpatched vulnerability, exposed secret,
or incident containing personal data.

Preferred channel:

1. Use GitHub **Report a vulnerability** on the repository Security tab when
   private vulnerability reporting is available.
2. Otherwise email **aoi@aimojitech.com** with the subject
   `[KinaBot Security] Brief description`.

If email encryption is required, first request a suitable secure exchange
method without including exploit details or sensitive data. Never send real
participant audio, full transcripts, passwords, API keys, access tokens, or
production database exports as proof. Use redacted logs and synthetic samples.

Include, when available:

- affected version, commit, component, and deployment mode;
- vulnerability type and realistic impact;
- minimal reproducible steps or proof of concept;
- prerequisites and whether exploitation is remote or authenticated;
- evidence of possible exposure of audio, transcripts, reports, identity,
  consent records, secrets, or research data;
- suggested mitigation and any disclosure deadline;
- how you would like to be credited, or whether you prefer anonymity.

## What to Expect

Maintainers will make a good-faith effort to:

- acknowledge a complete report as capacity allows;
- confirm whether the issue is reproducible and in scope;
- communicate material changes in status;
- coordinate a fix and public advisory when appropriate;
- credit the reporter with permission.

Response and remediation times depend on severity, reproducibility, maintainer
capacity, and third-party dependencies; this volunteer-stage project does not
promise a fixed service-level agreement. Please allow a reasonable private
coordination period before public disclosure. If users face active harm, data
exposure, or credential compromise, prioritize containment and notify affected
operators through the safest available channel.

## In-Scope Examples

- unauthorized access to accounts, recordings, transcripts, reports, or exports;
- bypasses of authentication, consent, retention, or deletion controls;
- temporary audio or transcript persistence after success or failure;
- predictable participant identifiers or broken pseudonymization;
- injection, path traversal, unsafe upload handling, or remote code execution;
- exposed secrets, insecure AWS configuration, or overly broad permissions;
- dependency vulnerabilities with a demonstrated KinaBot impact;
- privacy failures that disclose or silently retain sensitive data.

## Usually Out of Scope

- reports based only on automated scanner output without a reproducible impact;
- denial-of-service requiring unrealistic resources against a local prototype;
- social engineering, physical access, or attacks on unrelated third parties;
- missing hardening headers with no demonstrated security consequence;
- vulnerabilities that affect only unsupported historical code;
- clinical-effectiveness, diagnosis, or regulatory claims—KinaBot makes no such
  claims, though misleading product wording may still be reported publicly as a
  safety issue when it contains no exploit or private data.

These examples guide triage and do not prevent maintainers from investigating a
credible good-faith concern.

## Safe-Harbor Intent

The project supports good-faith research that avoids privacy violations,
service disruption, data destruction, persistence, extortion, and access beyond
what is necessary to demonstrate the issue. Stop testing and report immediately
if you encounter personal data, credentials, or evidence of active compromise.

This statement expresses project intent and is not legal advice or authorization
to test systems, accounts, data, or infrastructure you do not own or have
explicit permission to assess.

## Responsible-Use Boundary

KinaBot must not be used for surveillance, hidden recording, coercive
monitoring, employment or insurance decisions, diagnosis, or other uses that
undermine dignity, privacy, consent, or autonomy. Security review does not make
the software clinically validated or suitable for high-risk decisions.
