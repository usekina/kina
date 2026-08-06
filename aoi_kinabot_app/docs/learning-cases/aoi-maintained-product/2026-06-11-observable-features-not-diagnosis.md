# Replacing Diagnostic-Like Scores with Observable Features

**Date:** 2026-06-11

**Status:** Implemented product boundary

**Source:** V1 plans, scoring design, implementation, and tests

## Background and Problem

Speech technology can produce numbers that look authoritative before their
meaning is scientifically established. Early exploratory concepts included
composite or risk-style ideas. For a human-centered wellness product, those
presentations could cause a person to interpret a short voice sample as a
judgment about intelligence, age, disease, or personal worth.

The engineering challenge was not only to add a disclaimer. The system needed
an architecture whose outputs matched the modest claim it could responsibly
make.

## Alternatives Considered

- Keep an attractive overall score and explain its limitations.
- Produce disease or cognitive-risk estimates while labeling them
  experimental.
- Use an LLM to infer a holistic condition from audio or transcript.
- Present separate, versioned speech and language features for reflection over
  time.

## Decision and Implementation

Aoi selected the fourth approach. The maintained V1 separated observable
features, documented their transformations, versioned the scoring behavior,
and excluded cognitive age, dementia risk, diagnosis, and treatment advice.
Temporary audio handling, consent, data access, pilot planning, database
helpers, and tests were developed with that boundary.

## Maintainer Reflection

> I learned that responsible AI is not achieved by placing a warning beneath an
> overreaching score. The claim boundary has to shape the data model, scoring
> functions, interface language, retention policy, and tests. A less dramatic
> result can be more useful when a person can understand what it does and does
> not mean.

## Reusable Knowledge

- A precise feature is preferable to an unsupported composite label.
- Interface tone cannot repair a method that exceeds its evidence.
- Store raw measurements and method versions so later changes remain auditable.
- Treat non-diagnostic positioning as an engineering constraint, not marketing
  copy.

## Verification and Open Evidence

The 2026-06-11 sequence includes the feature-score design (`2c4f46d`), local
verification helpers (`e1959b9`), scoring functions (`94c305a`), and application
skeleton (`8c2f21a`). See
[`SCORING-METHODOLOGY.md`](../../../SCORING-METHODOLOGY.md) and
[`METRIC-DEFINITIONS.md`](../../methodology/METRIC-DEFINITIONS.md).

These records show a reproducible design decision. They do not establish that
the features are clinical biomarkers or that they predict health outcomes.

## Discussion Questions

1. When does a numerical score become a health claim?
2. Can a disclaimer compensate for a misleading product architecture?
3. What evidence would be required before adding a clinically meaningful
   interpretation?
