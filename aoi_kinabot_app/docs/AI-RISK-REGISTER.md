# KinaBot AI Risk Register

Status: Active V1.1 register

Risk owner: Aoi Minamoto / AImoji LLC

Last reviewed: 2026-08-05

## Rating Method

Likelihood and severity are rated from 1 (low) to 5 (high). Inherent risk is
the product before controls; residual risk is the expected risk after current
controls. Ratings are decision aids, not quantitative predictions.

Any risk with residual severity 5, residual score 12 or higher, evidence of
actual harm, unauthorized sensitive-data disclosure, or erosion of the
non-diagnostic boundary blocks release until reviewed and accepted in writing.

## Active Register

| ID | Risk and affected people | Inherent L×S | Current controls | Residual L×S | Evidence needed / next action | Status |
|---|---|---:|---|---:|---|---|
| R-01 | A user interprets a feature score as diagnosis, disease risk, intelligence, or cognitive age | 4×5 | Non-medical language; separate feature descriptions; disclaimers; retired composite risk outputs | 2×5 | Comprehension testing with older adults; revise any misunderstood screen | Open |
| R-02 | Ordinary variation, topic, fatigue, or mood is mistaken for meaningful decline or improvement | 4×4 | Personal history language; causal and health claims prohibited | 3×4 | Reliability study, confidence intervals, minimum detectable change | Open—release limiting |
| R-03 | Transcription error creates misleading language features | 4×4 | Private transcription; transcription-clarity feature; no diagnosis | 3×4 | Language/device error analysis and reject-low-quality gate | Open—release limiting |
| R-04 | Device, microphone, or noise changes distort pace and pause trends | 4×3 | Timestamp calculations and session metadata | 3×3 | Device/noise robustness study; record device/quality metadata | Open |
| R-05 | English-derived assumptions disadvantage Japanese, Chinese, dialect, or bilingual users | 4×4 | Language-specific adapters and reference centers; no cross-language comparison | 3×4 | Fluent review, subgroup analysis, bilingual/code-switch tests | Open—release limiting |
| R-06 | A family member monitors or coerces an older adult | 3×5 | User account and consent framing; family sharing is not automatic | 2×5 | User-controlled sharing permissions, revocation, coercion-sensitive UX study | Open—release limiting |
| R-07 | Sensitive identity, audio, transcript, or longitudinal data is exposed | 3×5 | Temporary audio; transcript not persisted; separated exports; secrets handling | 2×5 | Threat model, access audit, deletion verification, incident exercise, independent security review | Open—release limiting |
| R-08 | Product consent is reused as research consent | 3×5 | Documentation explicitly separates product and research use | 1×5 | Separate research workflow and ethics/IRB determination before research | Controlled; monitor |
| R-09 | Optional generative AI invents health meaning or inappropriate advice | 3×5 | LLM does not score; anonymous trend payload; curated actions; explicit prohibitions | 2×5 | Adversarial prompt tests, structured-output validation, kill switch, output audit | Open—release limiting |
| R-10 | Wellness advice is misunderstood as treatment or as a way to repair a score | 3×4 | General-action library; sources support habits only; non-treatment boundary | 2×4 | Comprehension tests and professional content review | Open |
| R-11 | Accessibility barriers exclude older adults or users with disability | 4×4 | Mobile-first design; upload flow; plain-language intent | 3×4 | Keyboard, screen-reader, contrast, motor, hearing, and low-literacy testing | Open—release limiting |
| R-12 | Engagement design creates guilt, compulsion, or repetitive testing | 3×3 | One recommended reflection; optional extras; daily maximum; no streak penalty | 1×3 | Monitor repeated attempts, distress feedback, and withdrawal | Controlled; monitor |
| R-13 | A security or reliability failure is not detected or handled promptly | 3×5 | Health checks, logs, deployment documentation | 2×5 | Incident response plan, alert ownership, backup/restore and rollback exercises | Open |
| R-14 | Algorithm update breaks longitudinal comparability | 4×4 | Scoring version stored; migration required for mixed-version trends | 2×4 | Automated compatibility tests and formal version-change review | Open |
| R-15 | Small or unrepresentative pilot data is used to make broad claims | 4×4 | Verifiable impact rules; no clinical claims | 2×4 | Preregistered analysis, sample-size rationale, transparent limitations | Open |
| R-16 | Public metrics or testimonials expose identity or overstate benefit | 3×4 | Aggregate-only impact record; no private production exports in GitHub | 1×4 | Evidence review and privacy check before publication | Controlled; monitor |
| R-17 | Founder, contributor, or invention claims misstate another person's work | 3×4 | Ownership record distinguishes roles; Git/PR history retained | 1×4 | Contributor agreements and claim-by-claim counsel review where applicable | Controlled; monitor |
| R-18 | Service becomes financially or operationally unsustainable | 3×3 | Usage limits; local scoring; container/AWS direction | 2×3 | Cost per active user, support burden, uptime, and funding model tracking | Open |
| R-19 | Users rely on KinaBot during urgent health or safety concerns | 2×5 | Not emergency support; professional-care direction | 1×5 | Test visibility and comprehension of urgent-use boundary | Controlled; monitor |
| R-20 | Environmental cost grows without corresponding public value | 2×3 | Local deterministic scoring; limited optional LLM use | 1×3 | Track compute per completed reflection and avoid unnecessary inference | Open |

## Release Review

For every material release, the owner must record:

1. risks changed or introduced;
2. user groups newly affected;
3. evidence reviewed;
4. tests and human review completed;
5. unresolved limitations;
6. rollback or disable mechanism; and
7. release, restrict, delay, or retire decision.

## Incident and Learning Loop

Reports of misunderstanding, distress, coercion, privacy failure, unsafe advice,
accessibility exclusion, or unexplained measurement behavior must be logged even
when they do not meet a legal definition of incident. The response record should
include detection date, affected versions, immediate containment, root cause,
corrective action, verification, user communication, and whether public
documentation needs correction.

The register is expected to evolve. Closing a risk requires supporting evidence;
a product decision alone does not prove the risk has been eliminated.
