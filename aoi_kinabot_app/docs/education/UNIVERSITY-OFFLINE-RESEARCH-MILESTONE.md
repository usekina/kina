# University Offline Research Mode — Data Science Milestone

Date established: 2026-08-05

## Milestone Purpose

This milestone records the development of a reproducible offline research mode
for two educational and research contexts:

1. a prospective UK university researcher who requested local/offline KinaBot
   use; and
2. the Purdue University Online Data Science educational record planned for the
   fall academic term.

No university endorsement, formal partnership, ethics approval, or deployment
is implied unless supported by a separate institutional document.

## Data Science Problem

Create a longitudinal multilingual speech-feature workflow that can operate
without cloud transcription, email identity, or generative-AI dependence while
preserving reproducibility and research governance.

## Engineering Deliverables

- school-assigned pseudonymous participant IDs, including values such as `001`;
- one-way domain-separated account keys with no raw ID stored in KinaBot;
- local SQLite longitudinal records;
- mandatory preinstalled local Whisper model;
- offline-only dependency manifest and launcher;
- code-level disabling of SMTP and OpenAI paths;
- de-identified research export;
- UK/EU data-protection readiness checklist;
- offline acceptance procedure; and
- automated tests covering identity, model, and external-service boundaries.

## Learning Outcomes

Students should be able to explain and evaluate:

- pseudonymisation versus anonymisation;
- data minimisation and purpose limitation;
- reproducible software and model versioning;
- offline dependency packaging;
- longitudinal data design;
- multilingual measurement limitations;
- separation of deterministic analytics from generative AI;
- privacy, fairness, and human-centered risk controls; and
- the difference between engineering readiness, research ethics approval, and
  legal compliance.

## Evidence to Preserve

- originating researcher request, with permission and appropriate redaction;
- dated issue/design record and Git branch;
- commits, pull request, reviews, and test results;
- exact offline package manifest and file hashes;
- demonstration screenshots or recording using synthetic/non-sensitive data;
- university feedback and requested changes;
- ethics/DPO decisions if a study proceeds;
- independent installation and acceptance result; and
- subsequent educational use, citation, or adoption records.

## Completion Criteria

The engineering milestone is complete only when a clean test environment can:

1. install from a local wheelhouse;
2. start with all network access disconnected;
3. accept participant ID `001` without email;
4. transcribe a non-sensitive sample from a local model;
5. calculate and persist feature results;
6. display longitudinal history;
7. export de-identified records;
8. demonstrate that OpenAI and SMTP were not called; and
9. pass the documented privacy and file-retention inspection.

Institutional deployment remains a separate milestone requiring university
approval and independent acceptance.
