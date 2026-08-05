# KinaBot Impact Metrics and Evidence Rules

Status: V1.1 measurement specification

Owner: Aoi Minamoto / AImoji LLC

Last updated: 2026-08-05

## North-Star Outcome

KinaBot exists to help a person understand their own speech patterns over time
and prepare clearer, self-directed conversations with trusted family,
caregivers, or professionals while preserving dignity, privacy, and autonomy.

The proposed primary real-world outcome is:

> **Supported conversation rate:** the proportion of eligible pilot participants
> who report that KinaBot helped them prepare or conduct at least one clearer,
> self-directed conversation during the observation period.

This is a communication and reflection outcome, not a medical outcome.

## Metric Hierarchy

### 1. Human Value

| Metric | Definition | Evidence source | Guardrail |
|---|---|---|---|
| Supported conversation rate | Participants answering a predefined positive response after a qualifying conversation ÷ eligible respondents | Preregistered participant survey | Report nonresponse and wording; do not imply health improvement |
| Reflection usefulness | Participants rating the experience useful for self-reflection | Standardized survey | Publish neutral/negative responses too |
| Perceived autonomy | Participants reporting control over recording, viewing, sharing, and stopping | Survey and interview | Investigate differences between user and caregiver responses |
| Family communication quality | Predefined change in a validated or justified communication measure | Prospective protocol | No causal claim without appropriate design |
| Caregiver burden | Change in predefined burden measure | Prospective protocol | Product must not shift hidden labor to families |

### 2. Safety, Dignity, and Understanding

| Metric | Definition | Target direction |
|---|---|---|
| Non-diagnostic comprehension | Users correctly explaining that KinaBot does not diagnose or predict disease | Increase; define gate before broad pilot |
| Score comprehension | Users correctly explaining what 0–100 feature indexes do and do not mean | Increase |
| False reassurance / unnecessary alarm | Users reporting unwarranted certainty caused by an output | Minimize and investigate every credible report |
| Coercion or unwanted monitoring reports | Reports that another person pressured or monitored the user | Zero tolerated without review |
| Privacy request completion | Access/deletion requests completed within documented service target | 100% within target |
| Adverse experience rate | Distress, misunderstanding, unsafe action, exclusion, or privacy complaint ÷ exposed users | Minimize; publish definition and follow-up |

### 3. Technical Trustworthiness

| Metric | Definition | Required breakdown |
|---|---|---|
| Analysis success rate | Valid results ÷ initiated eligible analyses | language, device, audio quality |
| Transcription error | Word or character error against reviewed references | language, age range, device, noise |
| Rejected sample rate | Samples blocked for insufficient quality ÷ uploads | language and device |
| Test-retest reliability | Preregistered reliability statistic for repeat conditions | feature and language |
| Minimum detectable change | Smallest change exceeding measured repeat variability | feature and language |
| Cross-version comparability | Sessions safely comparable under documented migration ÷ attempted comparisons | scoring-version pair |
| Accessibility task success | Participants independently completing core tasks | access need and device |

### 4. Sustainable Engagement

| Metric | Definition |
|---|---|
| Activation | Verified users completing a first valid reflection within the defined onboarding window |
| Three-reflection completion | Activated users completing three valid sessions |
| 7/30/90-day retention | Cohort members completing at least one valid reflection in the specified return window |
| Reflection-day completion | Distinct days with a valid reflection during the 30-day experience |
| Voluntary additional check-ins | Extra check-ins after the first daily reflection; monitored for compulsion, not maximized |
| User-initiated sharing | Users choosing to share an understandable summary; never automatic |
| Withdrawal rate and reason | Users stopping or withdrawing consent, including burden and privacy reasons |

### 5. Operational Sustainability

| Metric | Definition |
|---|---|
| Cost per completed reflection | Allocated infrastructure and service cost ÷ valid completed reflections |
| Support minutes per active user | Human support time ÷ active users |
| Service availability | Successful eligible requests ÷ total eligible requests during defined period |
| Recovery performance | Measured restore and rollback time during exercises/incidents |
| Compute per reflection | Transcription, scoring, and optional LLM resource use per valid reflection |
| Institutional continuation | Pilot institutions extending or renewing after documented review |

### 6. Independent Reach and Field Contribution

These metrics support public-impact and professional-contribution assessment but
must remain separate from product health claims.

| Metric | Acceptable evidence |
|---|---|
| Independent institutional adoption | Agreement, deployment record, and named independent contact |
| Independent replication | External protocol, code, report, or publication |
| Research citation | Verifiable scholarly citation that discusses or uses the work |
| Method reuse | External repository, curriculum, product, or protocol adopting a defined KinaBot method |
| Invited professional presentation | Organizer invitation and event record |
| Independent media coverage | Editorially independent publication focused substantially on the work |
| Standard or policy influence | Published standard, guideline, consultation, or policy record citing the contribution |

GitHub stars, internal tests, company-authored claims, paid publicity, and
solicited testimonials without corroboration are not substitutes for independent
adoption or broader professional significance.

## Cohort and Calculation Rules

Every reported product metric must specify:

- numerator and denominator;
- eligibility and exclusion rules;
- observation window and timezone;
- cohort start/end dates;
- software, consent, and scoring versions;
- missing-data and withdrawal handling;
- overall and relevant subgroup results;
- confidence interval where appropriate; and
- source export or signed record location.

Registrations, verified users, activated users, active users, sessions,
reflection days, and institutions must never be used interchangeably.

## Publication Rules

1. Preserve a dated private source snapshot before publishing an aggregate.
2. Require a second-person calculation and claims review for material results.
3. Suppress small cells that could expose identity.
4. Publish unfavorable and inconclusive outcomes with the favorable outcomes.
5. Separate association from causation and engagement from benefit.
6. Correct errors publicly and retain the correction history.
7. Never purchase or fabricate adoption, citations, awards, testimonials, or
   media evidence.

## Initial Dashboard: Current Status

| Category | Current status on 2026-08-05 |
|---|---|
| Automated engineering tests | Available; exact count must come from dated test run |
| Supported interface/analysis languages | English, Japanese, Chinese |
| Human-value outcomes | Not yet established |
| Reliability and minimum detectable change | Not yet established |
| Subgroup fairness | Not yet established |
| Independent institutional adoption | Not yet reported publicly |
| Independent replication | Not yet established |
| Clinical validity or health outcomes | Not claimed and not established |

The dashboard should remain honest when values are zero, unknown, negative, or
not yet measured. Its purpose is disciplined learning, not promotional scoring.
