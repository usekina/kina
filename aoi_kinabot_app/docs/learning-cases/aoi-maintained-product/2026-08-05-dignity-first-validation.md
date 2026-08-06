# Dignity First: Changing Product Success and Evidence Standards

**Date:** 2026-08-05

**Status:** Product, governance, and documentation framework established

**Source:** Dignity-first redesign and validation-governance work

## Background

A healthy-aging product can unintentionally turn reflection into surveillance,
competition, or fear. Streaks can punish missed days. Scores can feel like
grades. Caregiver features can weaken the older adult's agency. Warm language
can still conceal unsupported health implications.

The project needed a definition of success that included how a person is
treated, not only whether the software produced an output.

## Decision

KinaBot adopted a dignity-first product identity centered on longitudinal speech
reflection, healthy aging, and family-centered care. The experience recommends
one short daily reflection, makes additional check-ins optional, avoids streak
penalties, explains features without diagnosis, and separates participant and
research-administration flows.

The same release established validation gates, an AI risk register, and
evidence-defined impact metrics. This connected product values to release and
claim discipline.

## Maintainer Reflection

> I learned that dignity must be visible in small technical choices: whether a
> missed day is punished, whether a caregiver sees more than the participant
> expects, whether a score looks like a diagnosis, and whether uncertainty is
> stated before promotion. Human-centered technology is a continuing governance
> practice, not a friendly visual style.

## Reusable Knowledge

- Translate values into interface behavior, permissions, metrics, and release
  gates.
- Measure burden, comprehension, accessibility, and user agency—not engagement
  alone.
- Keep engineering verification, usability evidence, and clinical validation
  as distinct layers.
- Record risks before external attention makes them inconvenient to disclose.

## Evidence and Limits

The work is recorded in PR
[#28](https://github.com/usekina/kina/pull/28),
[`RESPONSIBLE-WELLNESS-DESIGN.md`](../../RESPONSIBLE-WELLNESS-DESIGN.md),
[`AI-RISK-REGISTER.md`](../../AI-RISK-REGISTER.md),
[`VALIDATION-PLAN.md`](../../VALIDATION-PLAN.md), and
[`IMPACT-METRICS.md`](../../IMPACT-METRICS.md).

This framework demonstrates responsible preparation. It is not proof that users
experience dignity, that families benefit, or that the product improves health.
Those claims require consented studies and independent evidence.

## Discussion Questions

1. Which product metrics can conflict with dignity?
2. How should participant agency shape family-centered features?
3. What evidence separates an appealing responsible-AI framework from an
   effective one?
