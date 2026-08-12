# When a Product Limit Protects the User

**Date:** 2026-08-12  
**Type:** Public product-decision learning case  
**Audience:** Students, product teams, engineers, researchers, and responsible-AI practitioners

## The question

KinaBot recommends one short voice reflection per day and permits up to three
completed analyses per verified user per local day. The maintainer asked:

> Should KinaBot remove the daily limit?

Removing a limit may look more user-friendly. In a longitudinal wellness
product, however, unlimited repetition is not automatically more humane or
scientifically useful. The decision must balance agency, measurement quality,
repetitive behavior, research needs, reliability, cost, and sustainability.

## The decision

Keep **three successful analyses per local day** as the public-service default
during the early-pilot stage. Continue recommending one low-pressure daily
reflection; additional check-ins remain optional.

Do not apply one rule to every context:

| Context | Policy |
|---|---|
| Public free service | Three successful analyses per local day |
| Future paid individual service | A higher but still responsible fair-use allowance, supported by evidence |
| University research | Configurable under the approved study protocol |
| Offline/Private Research API | Configurable by the institution administrator |
| Internal quality testing | Separate high-limit environment, isolated from participant data and public metrics |

Three is an operational default, not a clinically meaningful number. It should
change when evidence supports a better policy.

## Why unlimited is not automatically better

Unlimited same-day testing may encourage users to repeat a reflection until a
preferred score appears. It can:

- turn reflection into score optimization or compulsive checking;
- make short-term noise look like longitudinal change;
- amplify practice, fatigue, topic, emotion, and recording-condition effects;
- increase cost and abusive automated traffic without improving evidence.

A limit can be responsible interaction design when its purpose is clear, its
effects are measured, and it does not block access to history or data rights.

## Dignity-first UX

The limit must not sound punitive or imply failure. A suitable message is:

> Today's reflection is complete. KinaBot is designed to observe patterns over
> time rather than encourage repeated testing. You can return tomorrow.

Missing a day should not break a streak or create guilt. Users must still be
able to review prior results and understand why the limit exists.

## Count outcomes, not attempts

Only a successfully completed, valid analysis should consume an allowance.
These events should not count:

- recording, upload, transcription, analysis, storage, or timeout failure;
- cancellation before completion;
- reviewing an existing result or trend.

Users should never lose quota because the product failed. Implementation should
use an atomic completion record or equivalent idempotent transaction so retries
cannot be counted twice. Tests should cover concurrent requests, retries,
partial failure, timezone changes, and local-midnight reset.

## Configuration is a policy contract

KinaBot already exposes `KINABOT_MAX_TESTS_PER_DAY`. A production-quality
configuration should also:

- document valid values and the public default;
- reject unsafe or malformed settings at startup;
- show administrators which policy is active;
- record policy changes in release and study documentation;
- keep public, research, and test records separated;
- require research settings to match the approved protocol.

Increasing a quota does not authorize a new medical use or relax consent,
privacy, deletion, security, or claims boundaries.

## Measure before changing the default

Review the decision with aggregate, privacy-preserving evidence:

- percentage of active users reaching the limit;
- blocked fourth-analysis attempts and their stated purpose;
- limit events preceded by technical failure;
- cost and latency per successful analysis;
- later-day return behavior;
- signs of repetitive score seeking;
- research protocols requiring same-day repeated measures.

If users reach the limit because failures are counted, fix reliability rather
than raising the quota. If a reviewed study needs repeated measures, configure
that study rather than changing the public service for everyone. Test broader
access only when legitimate use improves without increasing harm, poor-quality
data, or unsustainable cost.

## Market lesson

Free and unlimited are different promises. A service that cannot sustain
privacy controls, reliability, support, and maintenance eventually fails its
users. Responsible offerings may include a bounded public service, paid
convenience or support, and institution-managed private deployment.

The value is not unlimited testing. The value is trustworthy longitudinal
reflection, user control, responsible interpretation, and reliable operation.

## Classroom exercise

Design a one-page quota policy for a multilingual longitudinal-reflection
product. Specify:

1. the behavior the limit encourages or prevents;
2. policies for public, paid, research, offline, and test environments;
3. the exact event that consumes quota;
4. failure and retry behavior;
5. non-punitive user-facing copy;
6. privacy-preserving review metrics;
7. evidence required to change the default;
8. claims the policy does not authorize.

Debate: **Is more access always more user-centered?** Identify when removing
friction increases agency and when thoughtful friction prevents confusion,
harm, manipulation, or unsustainable operation.

## Public takeaway

> International product quality is not demonstrated by removing every limit.
> It is demonstrated when a limit has a clear purpose, respects the user,
> adapts to legitimate contexts, treats technical failure fairly, is measured
> after release, and changes when evidence changes.

## Evidence and related materials

- [KinaBot README](../../../README.md)
- [Decision Log](../../DECISION-LOG.md)
- [AI Risk Register](../../AI-RISK-REGISTER.md)
- [Impact Metrics](../../IMPACT-METRICS.md)
- [Product and UX Learnings](../../PRODUCT-UX-LEARNINGS.md)
- [From Feature Variation to Responsible Interpretation](2026-08-12-feature-variation-responsible-interpretation.md)
- [Dignity First](2026-08-05-dignity-first-validation.md)

## Claims boundary

This case documents a product and engineering decision. It does not establish
that the current limit improves health, prevents compulsive behavior, or is a
clinically validated intervention.
