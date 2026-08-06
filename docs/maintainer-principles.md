# Maintainer Principles

KinaBot is maintained as a dignity-first, consent-first, privacy-aware project for older adults, families, and caregivers.

## 1. Human Dignity First

KinaBot should support people without reducing them to a score, label, or diagnosis.

Outputs should be written in respectful, non-alarming language.

## 2. Consent First

KinaBot should only be used when all recorded or analyzed people have given appropriate consent.

Hidden recording, surveillance, coercive monitoring, or unclear consent flows are not acceptable.

## 3. Not A Medical Diagnosis Tool

KinaBot does not diagnose dementia, cognitive impairment, or any medical condition.

Features that imply medical diagnosis, medical risk prediction, or cognitive age estimation should not be added without clinical validation and appropriate review.

## 4. Privacy By Design

KinaBot should minimize data collection and retention.

Users should understand whether audio, transcripts, or reports are stored, deleted, or processed by external services.

## 5. Human Review Matters

KinaBot outputs should support human reflection and better conversations with care professionals.

The project should not encourage automated care decisions without qualified human review.

## 6. Responsible Roadmap

New features should align with KinaBot's safety, privacy, and dignity boundaries.

If a feature increases risk for users, families, or older adults, it should be delayed, redesigned, or rejected.

## 7. Data-Loss Actions Must Be Explicit

**Documented:** 2026-08-05

**Scope:** AImoji product design and implementation

> **AImoji founder principle:** 任何会导致用户数据丢失的操作，都必须有明确的按钮、明确的提示，并且不能依赖用户猜测。

Any user-facing action that can delete, replace, reset, or otherwise cause the
loss of user data must provide:

- a distinct and intentionally labeled control;
- a plain-language warning that identifies what data will be lost;
- a clear statement about whether recovery is possible; and
- confirmation proportionate to the severity and reversibility of the loss.

Destructive behavior must not be hidden behind navigation, an ambiguous icon, a
toggle, a timeout, or an action whose data-loss effect the user is expected to
infer. Defaults should preserve user data unless deletion is necessary for a
disclosed privacy or retention purpose.

Automatic lifecycle deletion, such as removing temporary audio after
processing, remains appropriate when it is part of the stated privacy design.
It must be disclosed before collection, applied consistently, and covered by
tests and operational records rather than presented as a user-initiated action.

## 8. Dignity Through Access and Awareness

**Documented:** 2026-08-05

**Scope:** AImoji product purpose, access, and evidence boundaries

People should have a fair opportunity to notice and understand meaningful
patterns in their health and everyday functioning, and to decide when to seek
support. Access to understandable information and appropriate evaluation should
not depend on income, insurance status, race, language, or geography.

AImoji products should reduce informational and practical barriers without
turning access into overclaiming. They must not present speech-derived patterns
as direct measurements of the brain or nervous system, replace qualified
clinical evaluation, or imply that awareness alone is diagnosis.

Human-centered technology should support longitudinal awareness, informed
choice, family communication when the individual wants it, and appropriate
professional follow-up while preserving privacy, agency, and cultural and
linguistic accessibility.

KinaBot may act as a longitudinal mirror by quantifying the direction,
magnitude, and variability of defined speech and language features over time
under a versioned method. A lower or more variable feature value is a
feature-level observation; it must not be presented as direct evidence of
cognitive, brain, or neurological decline.
