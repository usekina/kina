# KinaBot Product and Engineering Roadmap

Last reviewed: 2026-08-24

This roadmap communicates direction, not a promise of delivery, funding,
clinical effectiveness, institutional approval, or regulatory status. Priorities
may change as evidence, risk, maintainer capacity, and user needs change.

## Product North Star

Help people reflect on observable patterns in a speech sample over time while
protecting dignity, consent, privacy, accessibility, and honest interpretation.
KinaBot feature indexes are descriptive engineering measures, not health scores,
diagnoses, disease predictions, cognitive age, or evidence of improvement or
decline.

## Current Commitments

These are the areas the maintained application is expected to protect on every
change:

- explicit consent and understandable non-clinical language;
- minimal collection, temporary-audio cleanup, retention transparency, and
  participant deletion controls;
- versioned and explainable English, Japanese, and Chinese feature scoring;
- separation of incompatible scoring versions in longitudinal trends;
- mobile-first participant results and accessible interaction patterns;
- offline/private research workflows that do not silently call cloud AI;
- synthetic test data, regression coverage, and reproducible engineering checks;
- clear separation between engineering verification and user, institutional,
  clinical, or regulatory validation.

## Now — Reliability and Trust Foundation

Work may be selected from these priorities when it has a scoped issue and an
owner:

- close remaining authentication, consent, deletion, and retention gaps;
- strengthen automated privacy-boundary and failure-cleanup tests;
- improve accessibility checks for keyboard, screen reader, contrast, motor,
  hearing, language, and low-literacy needs;
- make scoring definitions, model versions, raw metrics, and migration behavior
  auditable;
- improve production health checks, structured logs, backup/restore evidence,
  dependency review, and rollback instructions;
- validate mobile upload and results journeys on supported browsers and devices;
- keep contributor, security, release, and responsible-use documentation current.

## Next — Evidence and Interoperability

These directions require discovery or validation before being treated as
committed implementation:

- moderated usability research with smartphone users, including older adults;
- transparent measurement of task completion, comprehension, accessibility,
  retention behavior, and failure recovery;
- documented export schemas and compatibility guarantees for approved research;
- calibration datasets and language-specific error analysis using properly
  authorized, governed data;
- a dedicated responsive research-admin experience if validated workflows
  justify it;
- community-maintained translations with native-speaker review and explicit
  limitations;
- sustainable release cadence, maintainer coverage, and contribution pathways.

## Explore — Evidence Required

Exploration does not imply approval or planned delivery:

- progressive web or native mobile delivery, only if mobile web cannot satisfy a
  validated user need and the added privacy/security surface has an owner;
- consent-based sharing with a chosen family member, caregiver, or professional;
- privacy-preserving aggregate research workflows;
- additional languages after language-specific tokenization, validation, and
  documentation resources are available;
- integrations that preserve user control, data minimization, deletion, and
  responsible interpretation.

## Not Planned Without a New Evidence and Governance Basis

- medical diagnosis, dementia or disease-risk prediction;
- cognitive-age, biological-age, impairment, or clinical-progression labels;
- treatment recommendations or replacement of healthcare professionals;
- hidden recording, surveillance, coercive monitoring, or passive monitoring;
- employment, insurance, eligibility, or other high-impact decisions;
- public leaderboards, interpersonal ranking, streak penalties, or shame-based
  engagement;
- default retention of raw audio or full transcripts;
- mixing incompatible score-model versions in a single continuous trend;
- claims of safety, accessibility, clinical value, or impact without evidence.

## How Work Enters the Roadmap

A roadmap proposal should include:

1. a defined user and evidence-backed problem;
2. measurable acceptance criteria and explicit non-goals;
3. privacy, security, accessibility, multilingual, migration, and maintenance
   analysis;
4. an owner and realistic validation method;
5. an issue or decision record that makes uncertainty and tradeoffs visible.

Priority is based on user harm or benefit, urgency, strategic fit, confidence,
effort, operational cost, and responsible-use risk. A merged pull request is not
proof of deployment or real-world impact.

## Status Language

- **Committed:** expected to be protected or delivered once explicitly scoped.
- **Next:** valuable candidate awaiting capacity, evidence, or implementation.
- **Explore:** a question to investigate, not a promise.
- **Not planned:** outside current boundaries unless the evidence and governance
  basis materially changes.

Use GitHub issues and pull requests for live implementation status. Update the
`Last reviewed` date and link supporting evidence when this roadmap changes.
