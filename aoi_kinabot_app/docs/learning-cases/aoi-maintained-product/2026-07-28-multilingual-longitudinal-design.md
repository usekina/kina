# Multilingual Design Is More Than Translation

**Date:** 2026-07-28

**Status:** Implemented in the maintained application

**Source:** Multilingual NLP, acoustic-feature, localization, and export work

## Trigger

A global speech-reflection product cannot assume that rules designed for
English remain meaningful in Japanese and Chinese. Word segmentation, grammar,
pace units, script, examples, and interface expectations differ. At the same
time, repeated sessions require stable versions so a later result can be
interpreted against the method used at that time.

## Competing Requirements

- Give users an experience in the language they speak.
- Keep calculations deterministic and inspectable.
- Avoid sending sensitive audio, transcripts, names, or email to a generative
  model.
- Preserve comparability without pretending every language is structurally
  identical.

## Decision

KinaBot implemented language-specific adapters for English, Japanese, and
Chinese; local transcription and feature calculation; timestamp-based voiced
duration and pauses; versioned feature records; localized explanations; and a
research export separated from direct identity.

The optional generative layer was limited to evidence-bounded communication
from minimized score trends and curated actions. It did not become the scoring
authority.

## Maintainer Reflection

> I learned that multilingual inclusion is architectural work. Translating
> labels is the visible last step; the deeper work is deciding what a token, a
> pause, a pace measure, and a fair comparison mean in each language. I also
> learned to keep generative fluency separate from measurement authority.

## Reusable Knowledge

- Internationalization and methodological localization are different tasks.
- Document language-specific assumptions and validation separately.
- Keep deterministic measurement independent from generated explanations.
- Version research-relevant behavior before collecting longitudinal data.
- Do not interpret equal-looking numbers as automatically equivalent across
  languages.

## Evidence and Limits

Public evidence includes PRs
[#16](https://github.com/usekina/kina/pull/16),
[#17](https://github.com/usekina/kina/pull/17), and
[#20](https://github.com/usekina/kina/pull/20), plus
[`ARCHITECTURE.md`](../../../ARCHITECTURE.md) and
[`ENGINEERING-LEARNINGS.md`](../../ENGINEERING-LEARNINGS.md).

Implementation in three languages is not evidence of cross-language validity,
measurement invariance, accessibility for all speakers, or equal performance
across accents and conditions. Those require representative evaluation.

## Discussion Questions

1. Which speech features can be shared across languages, and which require
   language-specific definitions?
2. How should a team report cross-language performance without hiding unequal
   sample sizes?
3. Where can generative AI add value without controlling the measurement?
