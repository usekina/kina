# Engineering Learnings

## 1. Separate Scoring from Explanation

KinaBot's versioned Python/NLP pipeline calculates the features. An optional LLM
may turn anonymous longitudinal score patterns and a curated action library into
plain language, but it is not the scoring authority.

This separation improves reproducibility, cost control, failure handling, and
privacy.

## 2. Design a Safe Fallback

External APIs can fail because of credentials, quota, model access, TLS
inspection, or network problems. KinaBot falls back to a deterministic,
research-linked action library so a user still receives a result.

The next operational improvement is explicit, privacy-safe observability that
distinguishes an AI-selected action from a fallback without logging sensitive
payloads.

## 3. Treat Audio as Ephemeral

The server needs a temporary copy to transcribe uploaded or browser-recorded
audio. The copy must be deleted on success and error paths. Raw audio and full
transcripts are not retained and are not sent to the LLM.

Persistent records contain account data, consent, session metadata, raw feature
metrics, scores, and scoring versions.

## 4. HTTPS Is a Product Requirement

Mobile browser microphone access requires a secure context. An HTTP load
balancer URL may serve a page successfully while the recording feature still
fails.

The pilot uses:

```text
Mobile or desktop browser
  -> CloudFront HTTPS
  -> Application Load Balancer
  -> ECS Fargate
  -> encrypted EFS-backed SQLite
```

CloudFront caching is disabled for the dynamic Streamlit application, and
viewer requests are forwarded to preserve sessions and interactive behavior.

## 5. Store UTC and Local-Time Meaning

UTC alone is insufficient for a daily-use product. KinaBot stores:

- `created_at`: timezone-aware UTC timestamp;
- `session_date`: browser-local calendar date; and
- `timezone_name`: IANA timezone used for that date.

Legacy sessions without a timezone are corrected once when the user returns.
New sessions retain their original timezone even if the user later travels.

## 6. Make Database Migrations Additive

The pilot database is persistent. New profile and timezone fields are added with
idempotent migrations rather than replacing the database. Deployment must never
erase EFS data.

SQLite is appropriate for a single-task pilot. Before horizontal scaling or
material concurrent use, migrate transactional data to a managed relational
database such as PostgreSQL.

## 7. Version Research-Relevant Behavior

Each result should retain:

- application version;
- scoring-model version;
- consent version;
- spoken language;
- session timestamp and local date;
- raw feature metrics; and
- displayed feature scores.

Without versioning, longitudinal comparisons become difficult to interpret
after an algorithm change.

## 8. Keep Identities Separate from Research Exports

The private user export contains direct identifiers and must be access
restricted. The research export uses a participant ID and excludes email,
display name, recordings, and transcripts.

De-identification alone does not create research consent or remove governance
responsibilities.

## 9. Test the Real Runtime

Local syntax checks are necessary but insufficient. Validate:

- the Docker image;
- database migration on persistent data;
- ECS service stability;
- HTTPS response;
- mobile microphone permission;
- three-language output;
- local-midnight behavior; and
- safe behavior when external services are unavailable.

## 10. Never Put Secrets in Git

API credentials belong in AWS Secrets Manager or local environment variables.
Logs and diagnostic commands should be designed so exceptions cannot echo a
secret. If a key appears in terminal output, revoke and rotate it.

## 11. Offline Is A System Property, Not A Label

Local scoring alone does not make a system offline. Identity, dependency
installation, model loading, telemetry, optional APIs, network binding, update
behavior, and acceptance testing must all be examined.

Short school IDs also show why context matters: an ordinary hash can be
enumerated, while a study-secret HMAC creates a stronger pseudonym but adds
secret backup and recovery responsibilities. See the dated case study,
[From Online Product to Offline University Research](learning-cases/aoi-maintained-product/2026-08-05-offline-university-research.md).
