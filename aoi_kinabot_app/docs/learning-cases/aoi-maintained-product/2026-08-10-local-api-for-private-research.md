# From an Offline App to a Private Research Capability

**Date:** 2026-08-10

**Maintained product:** `aoi_kinabot_app/`

**Product direction and decision owner:** Aoi Minamoto, AImoji LLC

**Evidence status:** implemented and automatically tested on the feature branch;
external university use and impact remain to be independently documented.

## Context

After an earlier request for offline KinaBot use, a UK university researcher
reported that an intern would use a Windows 11 university PC with an Intel i7.
The computer was not fully air-gapped, but the research context still required
a design that did not depend on email, cloud transcription, or external AI.

Aoi then asked whether another application could run KinaBot through an API.
The requirement was not merely machine-to-machine access. The integration had
to preserve KinaBot's dignity-first, non-diagnostic, privacy-aware,
longitudinal product boundary.

## Aoi's Product and Engineering Decisions

Aoi established that the university team should receive a practical, local
integration capability rather than a cloud dependency. She required that the
work be independently packaged, easy for a research team to operate, and
recorded as a high-quality engineering contribution.

The resulting decisions were:

1. Create a dedicated `offline_api/` subsystem while reusing KinaBot's existing
   transcription, multilingual analysis, scoring, identity, and database
   modules. This avoids a second, inconsistent scoring authority.
2. Bind the service to `127.0.0.1` by default. A local API does not need to
   become an internet service.
3. Require a locally generated research token for participant-data operations.
4. Convert a school ID such as `001` immediately into a study-secret HMAC
   pseudonym and never return or store the raw ID.
5. Remove `OPENAI_API_KEY` during startup and use only preinstalled local
   Whisper plus KinaBot scoring. University use therefore does not consume
   Aoi's OpenAI account or create an AImoji per-request API bill.
6. Return scoring and application provenance, retention behavior, `self_only`
   comparison scope, and `non_diagnostic: true` as part of the API contract.
7. Add an explicit participant deletion operation.
8. Preserve the distinction between engineering verification and claims of
   clinical validity, legal compliance, institutional approval, or impact.

## Why a Simple Cloud Endpoint Was Rejected

Sending university recordings to a public endpoint would introduce a new data
transfer, internet dependency, account relationship, availability risk, and
cost boundary. It would also weaken the earlier offline design. A local-only
API lets approved software integrate while raw data and computation remain on
the research computer.

## Implemented Evidence

- `offline_api/api.py`: FastAPI surface, authorization, file limits, health,
  analysis, history, and deletion endpoints.
- `offline_api/service.py`: reuse of the authoritative pipeline, HMAC identity,
  consent recording, versioned persistence, and non-diagnostic contract.
- `offline_api/run-offline-api.ps1`: localhost startup, local secrets, local
  Whisper, local database, and explicit OpenAI-key removal.
- `database.py`: pseudonymous lookup without accidental participant creation
  and deletion of participant records.
- `offline_api/tests/test_api.py`: authorization, non-disclosure, provenance,
  deletion, ID validation, and file-type tests.
- `offline_api/README.md`: installation, API, expense and governance boundaries,
  verification, and pilot-evidence guidance.

At implementation time, 27 automated tests passed: 18 existing multilingual
tests, six existing offline-mode tests, and three new API scenarios. The exact
result should be re-established by CI for the final commit.

## Human-Centered Significance

Interoperability can widen access, but it can also distribute risk. Treating
privacy, identity, deletion, interpretation limits, and provenance as part of
the interface means another application cannot integrate only the attractive
scores while silently omitting their intended meaning. The API turns the
dignity-first boundary into a machine-readable product contract.

## Business and Sustainability Lesson

No OpenAI bill does not mean no product value. The reusable value includes the
longitudinal method, multilingual engine, privacy architecture, validated
bundle, integration support, training, maintenance, institutional deployment,
and future partner services. A limited pilot does not commit AImoji to free
production support or permanent commercial rights.

## Claims Boundary and Next Evidence

This repository establishes dated implementation, reasoning, maintainership,
and automated engineering verification. It does not yet prove university
installation or adoption, participant benefit, clinical validity, or
field-wide impact.

Those claims require independent records: university acceptance, approved
study documents, deployment checksum, aggregate use, external feedback,
validation, publications, citations, continued adoption, and recognition.
Participant data and confidential correspondence must remain outside GitHub.

## Practical Takeaway

An offline API is not a contradiction. It is a stable software contract inside
a controlled environment. For sensitive human data, the strongest API may be
the one that gives other applications interoperability without giving the
internet the data.
