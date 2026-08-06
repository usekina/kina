# KinaBot Project Evolution and Continuing Contribution Record

This document preserves a factual, version-controlled account of KinaBot's
development. It distinguishes historical collaboration from the current
Aoi-maintained application and does not claim clinical validation or medical
effectiveness.

## 1. Origin and Early Public Exploration

KinaBot's product history traces its origin to the name Kizuna in May 2025.
The repository's complete public Git history across preserved refs begins on
November 28, 2025, with Aoi Minamoto's initial repository work and open-source
V1 publication. The current maintained branch contains a later reconstructed
history beginning in June 2026, so provenance checks should inspect all refs
rather than the current branch alone.

The early repository explored speech-derived cognitive-insight concepts,
Streamlit delivery, privacy and medical disclaimers, documentation, automated
testing, PDF reporting, and frontend/backend separation. Irene Li and Yuan Chen
made identifiable contributions during this collaborative period. Aoi Minamoto
founded the product, maintained the repository, integrated contributions, and
continued directing its public development.

These early materials are retained for provenance. Some include cognitive-age,
overall-score, or risk-style concepts that are not part of the current product.

## 2. Product Boundary and Responsible Redesign

In June 2026, Aoi established `aoi_kinabot_app/` as the maintained application
workspace and documented a clear boundary between historical experiments and
current product behavior. The redesign replaced diagnosis-like composites with
separate, observable speech and language feature indexes intended for personal
reflection over time.

The maintained design introduced:

- explicit founder, maintainership, authorship, and inventorship boundaries;
- consent-first use and responsible contribution rules;
- temporary audio handling and deletion procedures;
- a V1 pilot plan, implementation plan, feature-score design, and data-access
  model; and
- a non-diagnostic product boundary that excludes cognitive age, dementia risk,
  disease prediction, and treatment recommendations.

This was a substantive change in technical direction, not only a wording
change. The application architecture and data lifecycle were redesigned around
privacy, reproducibility, longitudinal reflection, and human review.

## 3. Multilingual and Longitudinal Engineering

During July 2026, the maintained application developed into a multilingual,
versioned speech-reflection system. The work included:

- English, Japanese, and Chinese language-specific NLP adapters;
- local Python/NLP feature calculation rather than LLM-generated scoring;
- locally controlled transcription using `faster-whisper`;
- timestamp-derived voiced duration and pause measurements;
- versioned feature records that support comparison across sessions;
- user-local date handling for longitudinal records and daily limits; and
- a data-minimized optional LLM insight layer that receives neither audio,
  transcripts, names, nor email addresses.

The core design decision was to keep the feature engine deterministic and
auditable while limiting generative AI to evidence-bounded, general wellness
communication.

## 4. Privacy, Research Readiness, and Deployment

The project then added a separation between de-identified longitudinal research
records and restricted identity/contact records. It documented consent,
retention, access, and export boundaries and established a research-admin view
without publishing production data.

AWS staging, HTTPS delivery, secrets handling, containerization, health checks,
and deployment documentation were added to move the project beyond a local
prototype. These engineering steps do not establish clinical efficacy; they
establish a more testable and operationally responsible platform for future
pilots and independent evaluation.

## 5. Human-Centered Product Evolution

User-experience work in late July and early August 2026 added:

- a multilingual landing and verification flow;
- returning-user profile restoration;
- mobile-first access to current results and longitudinal trends;
- public metric definitions, anonymized feedback records, and design decisions;
- removal of unreliable in-browser recording when observed behavior did not
  meet the reliability standard; and
- a low-pressure **30 Days to Know Your Patterns** experience with one
  recommended daily reflection, optional additional check-ins, and no streak
  penalty.

This stage consolidated the current product identity:

> **KinaBot is a dignity-first, privacy-aware AI system for longitudinal speech
> reflection, healthy aging, and family-centered care.**

## 6. Nature of Aoi Minamoto's Continuing Contribution

The repository record identifies Aoi Minamoto as the product founder and
current maintainer. Aoi directs the maintained application's product design,
privacy model, multilingual NLP architecture, longitudinal experience,
wellness-action boundaries, and deployment direction through AImoji LLC.

The continuing contribution is demonstrated by dated source control, merged
pull requests, release records, architecture documents, decision logs, tests,
and changes made in response to observed reliability and responsible-use risks.
Commit counts are evidence of continuity and authorship activity; they are not,
by themselves, evidence of scientific validity or field-wide significance.

## 7. Verification Sources

The following records should be read together:

- repository Git history and pull-request history;
- [`OWNERSHIP-AND-MAINTAINERSHIP.md`](../OWNERSHIP-AND-MAINTAINERSHIP.md);
- [`CHANGELOG.md`](../CHANGELOG.md);
- [`DECISION-LOG.md`](DECISION-LOG.md);
- [`ARCHITECTURE.md`](../ARCHITECTURE.md);
- [`SCORING-METHODOLOGY.md`](../SCORING-METHODOLOGY.md);
- [`IMPACT.md`](../IMPACT.md); and
- the V1 pilot, implementation, feature-score, and data-access design records.

Future claims of adoption, clinical utility, independent replication, or
field-wide impact must be supported by separate third-party evidence.

## 8. Offline University Research Mode

In August 2026, Aoi Minamoto developed a separate offline research path in
response to interest in university use. It removes email verification, accepts
school-assigned IDs, stores study-secret HMAC pseudonyms, disables SMTP and
OpenAI, and requires an existing local speech model so the application cannot
silently download one. The work includes local installation and launch scripts,
checksum-based bundle controls, an offline acceptance procedure, UK/EU
data-protection readiness documentation, and a university data science
milestone record. Institutional adoption or approval must be documented
separately if it occurs.
