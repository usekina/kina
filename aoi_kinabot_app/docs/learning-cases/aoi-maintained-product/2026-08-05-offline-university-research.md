# From Online Product to Offline University Research

Date initiated: 2026-08-05

Status: Engineering implementation merged; institutional review and complete
OS-specific delivery bundle remain pending

Source: Paraphrased request from a prospective UK university researcher

Context: The researcher had previously asked whether KinaBot could be used
offline. The current Aoi-maintained application had become locally scored and
privacy-aware, but its identity, dependency installation, and model-loading
assumptions still reflected an online product environment.

Public-record basis: GitHub PR #29, merge commit `754a58d`, automated tests,
design records, and the documents linked below

## Background

KinaBot began as an online speech-reflection product. A returning user entered
an email, received a verification code, and built a personal history. The core
feature engine and transcription were local to the KinaBot environment, while
email delivery and an optional, limited OpenAI insight layer could use external
services.

That architecture can be reasonable for an online pilot. It does not
automatically fit a university study conducted on an offline or air-gapped
computer. A different social and institutional environment changed the meaning
of otherwise ordinary design choices.

## Trigger

The prospective researcher wanted to use KinaBot offline. This created a more
specific engineering question:

> Can a university team install, identify participants, transcribe speech,
> calculate longitudinal features, export research data, and verify the system
> without relying on email, cloud AI, or an undisclosed network download?

The correct response required more than adding an `offline=true` label.

## Competing Requirements

| Requirement | Tension created |
|---|---|
| Longitudinal identity | Recognize repeat sessions without collecting email |
| Research confidentiality | School IDs may be short and easy to guess |
| No external transfer | SMTP, OpenAI, telemetry, and model downloads must not occur |
| Reproducibility | Fix and record model, dependency, code, and scoring versions |
| Ease of use | Avoid requiring researchers to reconstruct Python manually |
| Institutional control | University controls device, linkage key, retention, and approvals |
| Honest claims | Offline engineering is not automatic GDPR compliance or ethics approval |

## Alternatives Considered

### Continue using email with a locally displayed code

This works without SMTP but collects an identifier the study may not need. It
was rejected for the formal offline mode.

### Store the school participant ID directly

This is simple but makes the database immediately identifiable to anyone with
the participant list. It was rejected.

### Store `SHA256("001")`

This hides the visible value but is weak because an attacker can calculate the
small range of possible hashes. It was replaced with a study-secret HMAC.

### Keep external AI available when an API key exists

This would make offline behavior depend on configuration discipline. It was
rejected in favor of a code-level boundary that always selects the local action
library.

### Let `faster-whisper` load a model by name

This can trigger a first-run download. Offline mode instead requires an existing
local directory and stops with a clear error if the model is absent.

## Decision

KinaBot Offline Research Mode now:

- accepts a school-assigned ID such as `001`;
- stores a keyed, study-specific HMAC pseudonym rather than the raw ID;
- collects no participant email and bypasses SMTP;
- disables OpenAI even if an API key is present;
- requires a preinstalled local Whisper model path;
- binds its standard launcher to `127.0.0.1`;
- retains local SQLite history and de-identified research export; and
- records the delivery bundle's Git commit and SHA-256 manifest.

The university retains responsibility for the separate participant linkage
file, lawful basis, special-category-data analysis, ethics review, DPIA decision,
device security, retention, deletion, and incident response.

## Implementation

```text
School-assigned ID
  -> local validation
  -> HMAC(study secret, normalized ID)
  -> SQLite participant key

Approved audio file
  -> temporary local file
  -> preinstalled faster-whisper model
  -> versioned Python/NLP features
  -> local longitudinal record
  -> temporary audio and full transcript deleted

Offline trend
  -> local curated action library
  -> no SMTP or OpenAI path
```

Delivery materials include an offline requirements file, installer, launcher,
bundle builder, checksum manifest, verifier, quick start, research guide, and
UK/EU data-protection readiness checklist.

## What Failed or Changed

### Ordinary hashing was not sufficient

The first implementation used a domain-separated SHA-256 hash. Review identified
that IDs such as `001` have too little entropy. The design was corrected before
release to use a study-secret HMAC. This also created a new responsibility: the
secret must be backed up securely because losing it breaks longitudinal linkage.

### The official model download was blocked

An attempt to download the Whisper model from its official source was blocked by
TLS certificate verification on a controlled enterprise network. SSL
verification was not disabled. The lesson is that an offline application needs
an institution-approved model and dependency acquisition process, not an
insecure workaround.

### A health endpoint was not a full user test

The Streamlit health endpoint returned `200 ok`, but it did not create a user
session. A second test used Streamlit AppTest to open the app, enter `001`, and
continue. It confirmed that SQLite stored neither an email nor the raw ID.

## Verification

- Python compilation passed.
- 24 automated tests passed.
- Offline ID `001` AppTest login passed.
- SQLite contained no email or raw `001` value.
- Different study secrets produced different pseudonyms for the same ID.
- OpenAI remained disabled with a test API key present.
- Missing model configuration failed before reading audio.
- Four PowerShell delivery scripts parsed successfully.
- GitHub Actions passed.
- PR #29 was merged to `main` on 2026-08-06 UTC.

The complete disconnected transcription bundle remains unverified until an
approved OS-specific wheelhouse and model are tested in the target university
environment.

## Remaining Risks and Evidence Gaps

- Target OS, Python version, CPU, and air-gap policy are unknown.
- University ethics, DPO, information-governance, and security review are pending.
- Pseudonymised data may remain personal data; HMAC is not anonymisation.
- Model and dependency licenses require institutional review.
- Secret backup and recovery need an acceptance exercise.
- Independent installation and usability are not yet demonstrated.
- No institutional adoption, endorsement, or research outcome is claimed.

## Social and Human-Centered Meaning

Software requirements are not determined by code alone. Email may support
continuity in a consumer service while creating unnecessary collection in a
study. Cloud APIs may improve convenience while conflicting with an air-gap
protocol. A hashed identifier may appear private while remaining guessable.

Responsible data science requires contextual engineering: understand who
controls the system, who bears risk, what institution governs it, what data is
necessary, and how one technical control can create a new human or operational
responsibility.

## Student Exercise

Assume a university wants a 12-week study with 60 participants. Produce:

1. a data-flow diagram and inventory;
2. a threat model for IDs, audio, models, exports, and the study secret;
3. an Article 6/Article 9 question list for the university DPO without making a
   legal conclusion;
4. a retention and deletion schedule;
5. an offline acceptance plan;
6. a response to loss of the participant-key secret;
7. three accessibility risks; and
8. an approve, restrict, delay, or reject recommendation.

Distinguish implemented evidence, assumptions, and evidence still needed.

## Discussion Questions

- Is an offline system always more private than a managed cloud system?
- Who should hold the participant linkage file and HMAC secret?
- What happens when a participant requests deletion using only the school ID?
- When does speech data become health or biometric data?
- Can the records be reused for a second study?
- What evidence is needed before discussing health-related meaning?
- Is a failed secure download an obstacle or a safety signal?
- How does the design change for macOS, Linux, shared devices, or multi-site use?

## Evidence Links

- Pull request: https://github.com/usekina/kina/pull/29
- Merge commit: https://github.com/usekina/kina/commit/754a58deda8b4fc3866ef9bf186884d93271b22b
- [Offline Quick Start](../../../OFFLINE-QUICKSTART.md)
- [Offline Research Guide](../../OFFLINE-RESEARCH-GUIDE.md)
- [UK/EU Data-Protection Readiness](../../UK-EU-DATA-PROTECTION-READINESS.md)
- [Offline Bundle Manifest](../../OFFLINE-BUNDLE-MANIFEST.md)
- [University Offline Research Milestone](../../education/UNIVERSITY-OFFLINE-RESEARCH-MILESTONE.md)
- [AI Risk Register](../../AI-RISK-REGISTER.md)
- [Validation Plan](../../VALIDATION-PLAN.md)

## Reuse and Claims Boundary

This case documents one project's reasoning and implementation. It is not legal,
medical, cybersecurity, or research-ethics advice. It does not establish GDPR
compliance, clinical validity, university approval, adoption, educational impact,
or field-wide significance. Those conclusions require separate institutional
and independent evidence.
