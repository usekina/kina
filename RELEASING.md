# Releasing KinaBot

This document defines how maintainers turn reviewed changes into an auditable
release. It does not authorize deployment or replace the AWS operations guide.

## Versioning

Use semantic versioning for application releases:

- **MAJOR** for intentionally incompatible user, API, or persisted-data changes;
- **MINOR** for backward-compatible features;
- **PATCH** for backward-compatible fixes and documentation corrections that
  materially affect safe use.

Scoring-model identifiers are versioned separately from application releases.
Any incompatible scoring change must receive a new scoring-model identifier and
must not be silently mixed with older scores in trends.

## Release Readiness

Before tagging a release:

1. confirm all intended pull requests are merged and CI passes on `main`;
2. review open high-severity security, privacy, data-loss, and scoring-accuracy
   issues;
3. update `aoi_kinabot_app/CHANGELOG.md` under an explicit version and date;
4. document migrations, configuration or dependency changes, rollback, and AWS
   operational impact;
5. verify temporary-audio cleanup, consent, authentication, deletion, core
   scoring, history-version separation, and supported multilingual flows;
6. confirm documentation and responsible-use wording match runtime behavior;
7. use synthetic data for evidence and remove secrets and personal information;
8. identify contributors from merged pull requests and material issue reports.

## Release Notes

Release notes should be understandable without reading commit history and use
these sections when applicable:

- Highlights
- Added
- Changed
- Fixed
- Privacy and security
- Scoring and migration
- Accessibility and multilingual behavior
- Deployment and operator action required
- Known limitations
- Contributors

Do not claim deployment, accessibility conformance, institutional approval,
clinical validation, or measured impact unless the release links evidence that
supports the exact claim.

## Tag and Publish

1. Create an annotated tag such as `v1.2.0` from the reviewed `main` commit.
2. Generate a draft GitHub release and edit it into the structure above.
3. Link the changelog, migration instructions, relevant issues, and pull
   requests.
4. Publish only after another reviewer or the maintainer verifies the tag,
   artifacts, dependencies, and claims.
5. Record deployment separately. If deployed to AWS, capture the environment,
   release/tag, timestamp, health check, smoke tests, rollback point, and known
   limitations without exposing infrastructure secrets.

## Contributor Recognition

Thank code and non-code contributors for material, attributable work, including
bug reports, design, tests, documentation, translation, accessibility review,
privacy review, and research feedback. Ask before publishing a real name or
sensitive affiliation; GitHub usernames may be used for public GitHub activity.
Recognition must not imply endorsement of the entire project.

## Correction and Withdrawal

If a release contains a security, privacy, data-loss, or scoring-integrity risk,
prioritize user protection. Mark affected notes clearly, publish mitigation or a
replacement release, and document whether operators must rotate secrets,
migrate data, redeploy, or notify affected users. Do not rewrite release history
in a way that hides the incident or original behavior.
