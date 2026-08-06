# Removing a Feature When Reliability Was Not Good Enough

**Period:** 2026-07-29 to 2026-07-31

**Status:** Implemented correction

**Source:** Recording, deployment, and upload-flow commits

## Background

Direct browser recording appeared to be the most human-centered mobile path:
it removed the need to find a separate recorder and locate an audio file. The
feature was added with upload as a fallback, and HTTPS infrastructure was used
because mobile microphone access requires a secure context.

## What Changed

Real runtime behavior did not meet the reliability standard across the intended
environment. A feature that looked simpler in the interface created uncertainty
when browser, permission, format, and deployment behavior varied.

On 2026-07-31, the maintained flow changed to reliable upload-only audio rather
than keeping an unstable primary path merely because it was more impressive.

## Maintainer Reflection

> I learned that human-centered design is not the same as minimizing the number
> of taps in a mock-up. Reliability is part of dignity: a person should not be
> asked to repeat a sensitive reflection because the product chose novelty over
> dependable behavior. Removing my own feature was progress when the evidence
> showed it was not ready.

## Reusable Knowledge

- Test media capture on the real browser, device, HTTPS route, and deployment.
- Count failed and repeated attempts as user harm, not only technical errors.
- Preserve a dependable fallback before promoting a more convenient path.
- A public reversal can be stronger engineering evidence than silently keeping
  a weak feature.

## Evidence and Limits

The progression is visible in commit `8f10fb7` (direct recording), deployment
corrections including `6e626bc`, and commit `5ed90b5` (reliable upload-only
flow), merged through PR
[#23](https://github.com/usekina/kina/pull/23).

The record shows a reasoned product correction. It does not yet contain a
published device matrix, failure-rate dataset, or controlled usability study.
Future reintroduction of direct recording should require those acceptance
criteria.

## Discussion Questions

1. When should a team remove a convenient but unreliable feature?
2. Which device and browser conditions belong in an acceptance matrix?
3. How should negative results be documented without exposing user data?
