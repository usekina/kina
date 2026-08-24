# Contributing to KinaBot

Thank you for helping make KinaBot safer, clearer, more accessible, and more
reliable. Contributions are welcome in many forms: code, tests, documentation,
translations, accessibility review, privacy review, reproducible research,
issue triage, and product feedback.

KinaBot is a dignity-first, consent-first, privacy-aware project for speech and
language-based cognitive wellness reflection. Please read the responsible-use
boundaries below before proposing or implementing a change.

## Start Here

Active development is in [`aoi_kinabot_app/`](aoi_kinabot_app/). Files outside
that directory may be historical exploratory material unless the repository
documentation says otherwise.

Before starting substantial work:

1. Search existing [issues](https://github.com/usekina/kina/issues) and pull
   requests to avoid duplicating work.
2. Open or comment on an issue describing the problem, intended users, proposed
   outcome, risks, and measurable acceptance criteria.
3. Wait for maintainer alignment before investing in a large feature or changing
   scoring, data handling, consent, safety wording, dependencies, or architecture.

By submitting a contribution outside the protected Aoi-maintained application
boundary, you must read and affirmatively accept [`CLA.md`](CLA.md). Contact the
maintainer before submitting on behalf of a company or other legal entity.

Small documentation corrections and narrowly scoped bug fixes may go directly
to a pull request when the problem and solution are clear.

### Aoi-maintained application boundary

`aoi_kinabot_app/` is the independently maintained KinaBot product area. Only
Aoi, using an approved personal or AImoji project identity, may author and
commit changes in that directory. External contributors are welcome to report
issues, supply reproducible evidence, review behavior, and propose patches, but
accepted implementation must be independently reviewed and submitted by Aoi.

This boundary preserves product responsibility and authorship provenance. It
must not be used to erase or under-credit the person who discovered a problem,
designed a solution, supplied research, or provided review. Those contributions
should be credited in the issue, decision record, pull request, or release notes
with permission.

## Choose the Right Contribution

### Report a bug

Include:

- a short description of the observed and expected behavior;
- exact steps and the smallest safe example needed to reproduce it;
- browser, operating system, Python version, and relevant app mode;
- logs or screenshots with secrets and personal data removed;
- possible impact on scoring, privacy, consent, deletion, or accessibility.

Never attach real participant audio, transcripts, email addresses, credentials,
medical information, or other sensitive data to a public issue. Create synthetic
test data that reproduces the behavior instead.

### Propose a feature or improvement

Explain:

- the user and problem, not only the requested interface or technology;
- evidence that the problem exists and why the current workflow is insufficient;
- the smallest useful scope and explicit non-goals;
- acceptance criteria that another person can verify;
- privacy, safety, accessibility, multilingual, migration, and maintenance
  implications.

Broad ideas such as “build a mobile app” must be refined into an actionable user
journey and testable outcome before implementation.

### Report a security or privacy concern

Do not publish exploit details, credentials, personal data, or an unpatched
vulnerability in a public issue. Use GitHub's private vulnerability reporting
for this repository when available. If no private channel is visible, contact
the maintainer privately and disclose only the minimum information necessary
until a safe reporting path is agreed.

## Responsible-Use Boundaries

Do not add features, labels, marketing claims, or documentation that present
KinaBot as:

- a medical diagnostic tool;
- a dementia or disease-risk predictor;
- a cognitive-age or biological-age estimator;
- a treatment recommendation system;
- evidence of clinical improvement or decline; or
- a replacement for licensed healthcare professionals.

Feature indexes are descriptive engineering measures of a speech sample. A
change to scoring must document its definition, limitations, language rules,
tests, and versioning or migration impact. Historical trends must not silently
mix incompatible scoring versions.

Preferred contributions include:

- privacy-aware collection, retention, export, and deletion controls;
- accessibility and mobile-web improvements;
- clear, non-clinical interface wording and documentation;
- reliable, explainable feature-level scoring;
- honest trend visualization;
- tests, observability, and failure recovery;
- multilingual support with language-specific rules and stated limitations.

## Local Development

KinaBot currently targets Python 3.10 in continuous integration.

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the maintained Streamlit application:

```bash
cd aoi_kinabot_app
streamlit run app.py
```

Never commit `.env` files, cloud credentials, API keys, participant data,
generated local databases, or production exports. Use synthetic data in tests
and examples.

Use a personal email address or an email address you are explicitly authorized
to use for this project in Git author and committer metadata. Do not use an
unrelated employer, university, customer, or third-party domain. Pull requests
containing a blocked employer domain fail the contributor-identity check; amend
or recreate the affected commits before requesting review.

## Make a Focused Change

1. Fork the repository and create a branch from the latest `main`.
2. Use a short descriptive branch name, such as `fix/connector-boundaries` or
   `docs/contribution-guide`.
3. Keep each pull request focused on one problem. Avoid unrelated formatting,
   dependency, or generated-file changes.
4. Add or update tests for behavior changes and update documentation for
   user-visible or operational changes.
5. Preserve safe cleanup behavior: temporary audio must be deleted even when
   transcription, scoring, storage, or rendering fails.

Commit messages should explain the outcome in imperative language, for example:

```text
Fix connector matching at English token boundaries
Document scoring-version migration rules
```

## Test Before Opening a Pull Request

Run the same core checks used by continuous integration from the repository
root:

```bash
python -m pytest
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

Also run focused tests while developing, for example:

```bash
python -m pytest aoi_kinabot_app/test_sentence_complexity.py
```

If a check cannot run in your environment, explain why in the pull request. Do
not remove or weaken a test merely to make CI pass without documenting the
underlying behavior change.

## Pull Request Requirements

In the pull request description:

- link the related issue (`Fixes #123` only when merging should close it);
- explain what changed, why it changed, and what is intentionally out of scope;
- list automated checks and manual flows used for verification;
- include before/after screenshots for visible changes, with private data
  removed;
- identify scoring-version, migration, privacy, security, accessibility,
  multilingual, and AWS deployment impacts—or state that each is not applicable;
- update user, scoring, API, operations, or deployment documentation as needed;
- disclose each new dependency and explain why it is necessary.

Before requesting review, confirm that the change:

- does not add diagnosis-like claims or misleading certainty;
- does not store raw audio or transcripts by default;
- does not add sensitive-data collection without explicit purpose and consent;
- preserves deletion, retention, and failure-cleanup behavior;
- contains no secrets, personal data, or confidential material;
- includes regression tests for fixed bugs and acceptance tests for new behavior;
- keeps existing user workflows working unless the change is intentionally
  breaking and includes a documented migration path.

## Definition of Done

A contribution is complete when its agreed acceptance criteria are met, relevant
tests pass, documentation is current, safety and privacy effects are addressed,
and a maintainer has reviewed and merged it. Merge does not by itself prove that
a change has been deployed to AWS or validated with real users; deployment and
measured outcomes must be recorded separately when they are part of the scope.

Review is collaborative and may require revision. A maintainer may decline work
that is out of scope or introduces unacceptable privacy, safety, ownership,
maintenance, or responsible-use risk. Decisions should be explained in the
relevant issue or pull request. Because maintainer capacity varies, the project
does not promise a fixed response or merge time.

## Contribution Rights and Recognition

By contributing, you confirm that:

- you have the right to submit the contribution;
- it is your own work or is properly licensed for inclusion;
- it may be distributed under this repository's license; and
- it contains no confidential, proprietary, medical, or personal data that you
  lack permission to share.

The prospective rights grant and commercial-relicensing terms are defined in
[`CLA.md`](CLA.md); this summary does not replace that agreement.

We recognize non-code contributions as well as merged code. GitHub records issue
authors, pull-request authors, reviewers, and commits; release notes or project
documentation may also thank contributors for material work. Recognition does
not imply that a contributor endorses every part of KinaBot.

Please interact with empathy, respect different backgrounds and levels of
experience, critique ideas rather than people, and protect participant dignity.
Harassment, discrimination, doxxing, and disclosure of private information are
not acceptable.

## Current Maintainership

KinaBot is currently maintained by Aoi Minamoto through AImoji LLC. Maintainers
make final decisions on scope, safety boundaries, releases, and deployment while
aiming to explain those decisions in the relevant issue or pull request.
