# KinaBot Product Learning Archive

This library turns meaningful product changes into dated, first-hand educational
case studies. It is intended for students, researchers, open-source builders,
and responsible-AI practitioners who need to understand not only what changed,
but why the surrounding human, institutional, legal, and technical environment
made that change necessary.

## Publication Standard

A change belongs here only when it teaches a reusable lesson and has enough
primary evidence to distinguish observation from hindsight. Good candidates
include:

- a new user or institutional environment that changes system requirements;
- a privacy, safety, accessibility, fairness, or reliability conflict;
- a failed assumption or implementation that materially changes the design;
- a decision that creates measurable tradeoffs;
- a validation result that confirms or overturns a design belief; or
- an external request that leads to a generalizable engineering method.

Routine copy edits, dependency bumps, and minor refactors do not become cases
unless they reveal a broader lesson.

## Required Case Structure

Every case should include:

1. date and status;
2. anonymized source and context;
3. prior system and triggering event;
4. competing technical, human, institutional, and legal requirements;
5. alternatives considered and the selected decision;
6. implementation and data flow;
7. failure, correction, or changed assumption;
8. verification and unresolved evidence;
9. social and educational meaning;
10. student exercises and discussion questions; and
11. evidence links and claims boundaries.

## Evidence Rules

- Link issues, commits, pull requests, tests, and dated documents.
- Paraphrase external feedback unless quotation permission is recorded.
- Do not expose personal, health, research, employment, or confidential data.
- Distinguish a prospective inquiry from adoption or institutional approval.
- Preserve negative and inconclusive evidence.
- Do not claim legal compliance, clinical validity, educational impact, or
field-wide significance without independent support.
- Correct retrospective records transparently rather than silently rewriting
  history.

## How to Read This Archive

The cases follow product turning points, not a marketing release calendar. Each
case separates three things:

- **contemporaneous evidence**: commits, pull requests, tests, and dated notes;
- **maintainer reflection**: what Aoi learned from making and reviewing the
  change; and
- **open evidence**: what still requires users, institutions, researchers, or
  independent evaluators.

First-person reflections are retrospective interpretations of the linked
record. They should be updated when stronger evidence changes the lesson.

## Learn or Verify

- Use the [Open Curriculum](OPEN-CURRICULUM.md) to teach, self-study, or run a
  product-team workshop from the cases.
- Use the [Evidence Map](EVIDENCE-MAP.md) to trace lessons to commits, pull
  requests, code, tests, and governing documents.
- Use the [Journey Timeline](JOURNEY-TIMELINE.md) to see the complete sequence
  without turning every small change into a separate case.
- Use the [White Paper Roadmap](WHITEPAPER-ROADMAP.md) to prepare the September
  2026 publication without treating planned evidence as completed evidence.

## Case Template

```markdown
# Case title

Date:
Status:
Source:
Context:
Public-record basis:

## Background
## Trigger
## Competing Requirements
## Alternatives Considered
## Decision
## Implementation
## What Failed or Changed
## Verification
## Remaining Risks and Evidence Gaps
## Social and Human-Centered Meaning
## Student Exercise
## Discussion Questions
## Evidence Links
## Reuse and Claims Boundary
```

## Two Sources of Learning

### Collaborative Exploration

The original project area preserves work by all identifiable contributors. Its
lessons concern collaboration, attribution, exploratory prototypes, and the
decision to establish a clearer product boundary.

- [2025-05 to 2026-06 · Origin, Collaboration, and an Independent Product Boundary](collaborative-exploration/2025-05-to-2026-06-origin-collaboration-boundary.md)

### Aoi-Maintained Product

The cases below concern the new `aoi_kinabot_app/` direction. The current Git
history for this path attributes its commits to Aoi Minamoto. These cases record
Aoi's product, engineering, privacy, reliability, and governance learning.

Start with the [Journey Timeline](JOURNEY-TIMELINE.md) for a compact record of
all meaningful milestones. The cases below examine the largest learning turns.

- [2026-06-11 · Replacing Diagnostic-Like Scores with Observable Features](aoi-maintained-product/2026-06-11-observable-features-not-diagnosis.md)
- [2026-07-28 · Multilingual Design Is More Than Translation](aoi-maintained-product/2026-07-28-multilingual-longitudinal-design.md)
- [2026-07-29 to 2026-07-31 · Removing a Feature When Reliability Was Not Good Enough](aoi-maintained-product/2026-07-29-to-31-recording-reliability.md)
- [2026-08-02 · From User Feedback to a Data Product](aoi-maintained-product/2026-08-02-feedback-to-data-product.md)
- [2026-08-05 · Dignity First: Changing Product Success and Evidence Standards](aoi-maintained-product/2026-08-05-dignity-first-validation.md)
- [2026-08-05 · From Online Product to Offline University Research](aoi-maintained-product/2026-08-05-offline-university-research.md)
- [2026-08-05 · Users Must Never Have to Guess About Data Loss](aoi-maintained-product/2026-08-05-explicit-data-loss-controls.md)
- [2026-08-05 · Dignity Through Access and Awareness](aoi-maintained-product/2026-08-05-dignity-through-access-awareness.md)

## The Next Cases Must Be Earned

Real-world pilots, independent validation, institutional adoption, published
research, citations, and demonstrated benefit are not current achievements just
because they are desired. They become new cases only after dated third-party
evidence exists. Until then, they remain validation goals in
[`VALIDATION-PLAN.md`](../VALIDATION-PLAN.md) and
[`IMPACT-METRICS.md`](../IMPACT-METRICS.md).
