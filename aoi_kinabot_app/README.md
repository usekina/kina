# KinaBot — Aoi-Maintained Application

> **Current implementation and product direction maintained by Aoi Minamoto through AImoji LLC.**

> **KinaBot is a dignity-first, privacy-aware AI system for longitudinal speech
> reflection, healthy aging, and family-centered care.**

This folder contains the current KinaBot application. Files elsewhere in this
repository are retained as historical exploratory work and do not define this
implementation's architecture, scoring policy, privacy policy, or product
claims.

KinaBot is a multilingual speech and language reflection system. Its core value
is a versioned Python/NLP feature engine that calculates observable features
from each user's voice sample and helps that user reflect on personal patterns
over time. The experience is designed to preserve autonomy, make privacy
boundaries understandable, and support—not replace—family and professional
care conversations.

KinaBot is not a medical device. It does not diagnose a condition, calculate
disease risk, estimate cognitive age, or replace professional care.

## Simple User Experience

1. Enter an email and a six-digit verification code on the same page.
2. Choose English, Japanese, or Chinese.
3. Select a recording stored on the user's phone or computer.
4. Run one analysis and see that sample's feature scores.
5. After three or more sessions, see personal trends and changes.
6. Receive a small, practical daily wellness action when appropriate.

The pilot recommends one 60-second reflection per day within a low-pressure
**30 Days to Know Your Patterns** experience. Additional check-ins are optional,
with a technical maximum of three analyses per verified email per day. Missing a
day is not treated as failure. Users do not create or remember a password.

## Core NLP Work

KinaBot itself calculates the feature scores. OpenAI does **not** calculate
them.

The versioned NLP pipeline measures observable properties such as:

- vocabulary or expression variety;
- response development and amount of speech;
- sentence structure and organization;
- speaking pace;
- pause behavior;
- repetition patterns;
- emotional expression in language; and
- transcription clarity.

These are feature levels for one sample. A higher or lower value is not, by
itself, evidence of better or worse health. Trend insights compare a user with
their own prior samples across multiple sessions and preserve the scoring
version used for every result.

English, Japanese, and Chinese are the first supported languages. Future
languages must use language-appropriate segmentation, grammar features,
normalization, testing, and calibration rather than treating English rules as
universal.

See [ARCHITECTURE.md](ARCHITECTURE.md) and
[SCORING-METHODOLOGY.md](SCORING-METHODOLOGY.md).

Project records:

- [Open knowledge center](docs/README.md)
- [Changelog](CHANGELOG.md)
- [Verifiable impact](IMPACT.md)
- [Founder and maintainership](OWNERSHIP-AND-MAINTAINERSHIP.md)
- [V1.1 release checklist](RELEASE-CHECKLIST.md)

## Offline University Research Mode

KinaBot can run without email, SMTP, OpenAI, or cloud transcription. An approved
offline bundle uses a preinstalled local Whisper model, local SQLite storage,
and school-assigned participant IDs such as `001`. The raw ID is not stored;
KinaBot derives a study-specific HMAC pseudonym using a locally generated secret.

Start with [Offline Quick Start](OFFLINE-QUICKSTART.md). Research administrators
should also review the [offline research guide](docs/OFFLINE-RESEARCH-GUIDE.md),
[UK/EU data-protection readiness checklist](docs/UK-EU-DATA-PROTECTION-READINESS.md),
and [bundle manifest](docs/OFFLINE-BUNDLE-MANIFEST.md).

Offline engineering controls support—but do not replace—university ethics,
information-governance, legal, security, and study-protocol approval.

## OpenAI's Limited Role

OpenAI is an optional insight layer used only after KinaBot has calculated the
scores. A future request may contain minimum anonymous structured data such as:

```json
{
  "language": "Japanese",
  "sessions_compared": 4,
  "feature_changes": {
    "expression_variety": [72, 68, 61],
    "speech_pace": [64, 65, 64],
    "repetition_consistency": [74, 65, 58]
  }
}
```

The request must not contain raw audio, a full transcript, name, email, or
direct identifiers. OpenAI must never be the feature-scoring authority.

The resulting insight should be a practical action for ordinary daily life:

- talk with a friend or family member for 20 minutes tomorrow;
- take a safe walk and tell someone one story from the day;
- read for 10 minutes and summarize the main idea;
- contact a friend the user has not spoken with recently; or
- choose an appropriate Mediterranean-style meal.

Every action should say what to do, when, for how long, how often, why it may
support general cognitive wellness, and which research source supports the
general habit. It must not claim to treat a condition, repair a score, or prove
that a score change represents cognitive decline.

## Privacy and Data

The source recording remains on the user's phone or computer. When selected
for analysis, KinaBot creates only a temporary server-side working copy.

- Speech-to-text is processed locally/private to the KinaBot environment.
- Python/NLP feature calculation is performed by KinaBot.
- Temporary audio is deleted after processing, including error paths.
- Raw audio and full transcripts are not stored.
- Audio and full transcripts are not sent to OpenAI.

KinaBot stores the minimum data needed for accounts and longitudinal use:

- name and email;
- optional profile fields and consent version;
- session date, language, duration, and software/scoring versions;
- calculated raw feature metrics and feature scores; and
- optional self-reported wellness habit check-ins.

Public research or publication requires separate consent, governance,
de-identification, and ethics review as applicable.

## Evidence-Informed Wellness Actions

Advice is general wellness information, not medical advice. Sources used to
shape the action library include:

- [WHO cognitive-decline and dementia risk-reduction guidelines](https://www.who.int/publications/i/item/9789241550543)
- [Mediterranean diet trial — PubMed](https://pubmed.ncbi.nlm.nih.gov/23670794/)
- [Social engagement and cognitive function study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6778491/)

A source supports the general habit; it does not establish that the habit will
change an individual user's KinaBot score.

## Run on a Local PC

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, normally `http://localhost:8501`.
Local transcription uses `faster-whisper`. Timestamped speech segments are
also used to calculate voiced duration, pause count, mean/maximum pause, and
pause ratio. The first analysis may take longer
while its model is downloaded and loaded; later analyses reuse the model.

OpenAI is not required for transcription or NLP scoring. If the optional
anonymous insight layer is later enabled, keep its API key in a local
environment variable or AWS Secrets Manager—never in GitHub.

## Deployment Direction

## Research Admin And Data Export

When `KINABOT_ADMIN_KEY` is configured, a signed-in owner can open
**Research admin** from the app sidebar. The panel shows user/session counts
and provides two deliberately separate exports:

- `kinabot_research_YYYY-MM-DD.csv`: de-identified longitudinal records with
  participant IDs, session metadata, scoring version, raw metrics, and 0–100
  sample feature scores; and
- `kinabot_users_private_YYYY-MM-DD.csv`: the private identity/contact list,
  which must be stored separately with restricted access.

The research export does not contain email addresses, display names, raw
audio, or full transcripts. Do not upload either production export to GitHub.

The application is packaged for Docker and future 24/7 AWS hosting. Production
should use HTTPS, a continuously running container service, managed relational
storage, secrets management, email delivery, health checks, backups, and
centralized logs. It must remain isolated from existing applications.
