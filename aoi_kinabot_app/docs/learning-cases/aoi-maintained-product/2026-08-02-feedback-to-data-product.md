# Practical Learning Case: From User Feedback to a Data Product

**Date:** 2026-08-02

**Status:** Feedback translated into an implemented and documented change

**Source:** Anonymized external product feedback and PR #27

This case uses an anonymized KinaBot product iteration to show how qualitative
feedback becomes a testable data-product decision. It is written as reusable
practical knowledge for learners, researchers, engineers, and product teams.

## Learning Objectives

Students should be able to:

1. separate a user's observation from a proposed solution;
2. translate feedback into product and data requirements;
3. explain a derived feature without overstating scientific meaning;
4. design a mobile longitudinal view for multivariate data;
5. distinguish engineering validation from clinical validation;
6. document privacy boundaries and versioned scoring decisions; and
7. connect an issue, implementation, test, release, and measured outcome.

## Case Prompt

An external tester reports that a single analysis is smooth but asks how eight
indexes are produced, wants to revisit the latest result, finds portrait-mode
scrolling excessive, and expects caregivers or older adults to use a phone
rather than a laptop.

Student teams should propose:

- a mobile information architecture;
- a transparent explanation of the eight indexes;
- a longitudinal visualization that avoids implying diagnosis;
- success metrics for a usability test; and
- a privacy-preserving feedback record.

## KinaBot Design Decision

KinaBot implements a compact two-column latest-result grid, a top-level Trends
destination, a one-feature-at-a-time trend chart, recent/all history controls,
and a versioned scoring explanation. Research administration remains
permission-gated and outside the participant's primary flow.

## Maintainer Reflection

> I learned not to treat feedback as a list of requested features. The deeper
> task was to find the unmet need: a mobile user wanted to understand the
> current result, return to it, and see change over time without being buried in
> technical detail. One comment affected navigation, visualization, scoring
> explanation, privacy separation, and future validation questions.

## Reusable Knowledge

- Preserve the original observation separately from the proposed solution.
- Translate qualitative feedback into testable product and data requirements.
- Make methodology reachable without forcing it into the primary user path.
- Link feedback, decision, implementation, test, and remaining measurement gap.

## Discussion Questions

- Does a 0–100 scale create unintended health-score expectations?
- When should an engineering reference center be recalibrated?
- How many repeat sessions are needed before a trend is useful?
- Which changes require a new scoring-model version?
- What evidence would be needed before discussing cognitive change?
- How should accessibility testing include older adults without stereotyping
  them?

## Evidence Discipline

The public repository demonstrates chronology, reproducibility, and design
reasoning. It does not by itself establish adoption, educational impact,
clinical validity, or extraordinary professional impact. Course delivery,
independent feedback, usage results, permissions, and external recognition
should be retained separately as verifiable evidence.

Public evidence includes the anonymized
[feedback record](../../feedback/2026-08-mobile-results-navigation.md), the
[mobile architecture note](../../ux/MOBILE-FIRST-RESULTS.md), and PR
[#27](https://github.com/usekina/kina/pull/27).

## Follow-On Session

The dated learning case
[From Online Product to Offline University Research](2026-08-05-offline-university-research.md)
extends this teaching material from mobile product feedback to contextual data
engineering. It asks students to redesign identity, model distribution, network
boundaries, pseudonymisation, validation, and institutional responsibility when
the same application enters a privacy-sensitive university research setting.

