# Dignity Through Access and Awareness

**Date:** 2026-08-05

**Status:** AImoji product-purpose principle documented

**Source:** Aoi Minamoto's first-hand professional observation and reflection

## Observation

Aoi observed that some people lack health insurance or cannot readily access
cognitive-health evaluation. Financial constraints, race-related inequities,
geography, language, and uneven service availability can affect whether a
person receives understandable information or timely professional support.
Changes may remain unnoticed, and families may not know when a pattern began or
when a conversation about evaluation could be useful.

This is a personal professional observation and design motivation. It is not
presented here as a prevalence study or a measured causal finding.

## Human-Centered Question

Should every person have a fair opportunity to understand whether meaningful
patterns may be changing in their health, cognition, or everyday functioning,
and to decide what to do next?

AImoji's answer is that access to understandable information and appropriate
evaluation should not depend on income, insurance status, race, language, or
geography. Awareness supports dignity only when it also preserves agency,
privacy, cultural context, and freedom from unsupported judgment.

## Product Principle

> Human-centered technology should reduce barriers to longitudinal awareness
> and informed action without replacing diagnosis or claiming to measure what
> its evidence cannot establish.

For KinaBot, this means helping a person observe patterns in their own speech
and language expression over time. It does not mean claiming that a voice sample
directly reveals a change in the brain or nervous system.

## KinaBot as a Longitudinal Mirror

“以此为镜”—use it as a mirror—is the intended metaphor. Under a stable,
versioned method, KinaBot can quantify the direction, magnitude, and variability
of defined speech and language features across a period of time. It can help a
person see that a feature has moved lower, higher, or become more variable than
their own earlier pattern.

The mirror does not explain the cause. A lower feature value is not equivalent
to cognitive decline, and greater variability is not evidence of a brain or
neurological disorder. Recording conditions, language, illness, fatigue,
medication, stress, device differences, and scoring-version changes may all
affect the observed pattern.

## Product Requirements

- Make the primary experience understandable without specialist vocabulary.
- Support linguistic and cultural accessibility rather than word-for-word
  translation alone.
- Design for low-cost and geographically distributed access where feasible.
- Explain what each observation can and cannot mean.
- Show feature direction, magnitude, and variability over time without
  relabeling feature change as medical decline.
- Preserve the individual's control over family sharing and follow-up.
- Avoid diagnosis, cognitive-age, disease-risk, or neurological-change claims
  without appropriate independent evidence and review.
- Provide a clear path from awareness to qualified professional evaluation when
  concern exists.
- Measure whether access, comprehension, and follow-up are equitable across
  relevant groups instead of assuming that availability creates fairness.

## Maintainer Reflection

> I have seen people remain unaware of possible cognitive change because
> evaluation and understandable information were not equally accessible. I
> believe people have a right to live with dignity and to have a fair opportunity
> to understand meaningful changes in their own lives. Technology should help
> lower that barrier, but it must not replace one injustice with another by
> offering unsupported conclusions or taking away a person's agency.

## Tension to Preserve

Two harms must be addressed together:

1. **Under-access:** people may lack affordable, local, or language-accessible
   ways to notice patterns and seek help.
2. **Over-interpretation:** an accessible AI tool may falsely imply diagnosis or
   direct knowledge of brain and neurological change.

A dignity-centered product must resist both. Wider access is not responsible if
the information is misleading; scientific caution is not equitable if it is
used as a reason to ignore access barriers.

## Student Exercise

Design an awareness tool for a population facing financial, geographic, or
language barriers. Specify:

1. what the tool observes;
2. what it explicitly does not infer;
3. how users understand uncertainty;
4. how privacy and individual agency are preserved;
5. how a user can seek qualified follow-up; and
6. what disaggregated evidence would be needed to evaluate equitable access and
   benefit.

## Discussion Questions

1. Is access to a tool meaningful if professional follow-up remains
   inaccessible?
2. How can a product support family communication without weakening the
   individual's control?
3. Which metrics could reveal that a multilingual product remains inequitable?
4. How should a team discuss early awareness without creating fear or implying
   diagnosis?

## Evidence and Claims Boundary

The formal principle is recorded in
[`docs/maintainer-principles.md`](../../../../docs/maintainer-principles.md) and
the initial documented principle is commit
[`2075664`](https://github.com/usekina/kina/commit/2075664).

This case documents Aoi's motivation and the product requirements derived from
it. It does not establish population prevalence, clinical effectiveness,
equitable outcomes, or a human-rights determination in law. Those claims require
appropriate research, legal analysis, and independent evidence.
