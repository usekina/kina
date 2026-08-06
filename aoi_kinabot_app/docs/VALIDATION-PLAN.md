# KinaBot Validation Plan

Status: V1.1 engineering and usability validation plan

Owner: Aoi Minamoto / AImoji LLC

Last updated: 2026-08-05

## Purpose

This plan defines the evidence KinaBot must collect before expanding product
claims or deployment. KinaBot currently supports non-medical, longitudinal
speech reflection. It is not clinically validated and must not diagnose,
screen for disease, predict medical risk, or recommend treatment.

Validation answers four separate questions:

1. **Technical validity:** does the software calculate the intended observable
   feature consistently?
2. **Measurement reliability:** are repeated results sufficiently stable under
   defined conditions?
3. **Human understanding:** do older adults and families understand the output,
   uncertainty, privacy choices, and non-diagnostic boundary?
4. **Real-world value:** does use support reflection and family communication
   without disproportionate burden, anxiety, false reassurance, or exclusion?

Success in one area must not be presented as success in another. Passing unit
tests, for example, does not establish clinical validity or real-world benefit.

## Intended Use Under Evaluation

KinaBot helps an adult record short, natural speech samples, review observable
speech and language features, and reflect on personal patterns across time. The
user controls whether to share observations with trusted family, caregivers,
or professionals.

The validation program excludes medical diagnosis, cognitive screening,
dementia-risk prediction, clinical decision support, treatment selection, and
unsupervised monitoring by another person.

## Evidence Levels and Release Gates

| Level | Question | Minimum evidence | Gate |
|---|---|---|---|
| E0: Code correctness | Does implementation match the documented formulas? | Unit, integration, privacy-boundary, and regression tests | Required for every release |
| E1: Controlled reliability | Are features stable across repeat recordings and devices? | Preregistered technical protocol, repeat samples, error analysis, confidence intervals | Required before interpreting small changes |
| E2: Multilingual validity | Do English, Japanese, and Chinese adapters measure their stated constructs appropriately? | Fluent-speaker review, transcription error analysis, language-specific distributions and failure analysis | Required before comparative public claims |
| E3: Human factors | Can intended users understand and safely use KinaBot? | Moderated usability study including older adults and caregivers | Required before broad public pilot |
| E4: Real-world utility | Does the product create useful reflection or communication value? | Prospective pilot with predefined outcomes and adverse-experience monitoring | Required before benefit claims |
| E5: Independent validation | Can an unaffiliated group reproduce findings? | Independent protocol execution, publication, or institutional report | Required for claims of broader professional significance |

No release may use evidence from a lower level to imply a higher-level result.

## Technical Validation Matrix

| Area | Primary measures | Required stratification | Initial acceptance rule |
|---|---|---|---|
| Transcription | word/character error rate; unusable transcript rate | language, device, noise condition, age range | Report results and failure modes; set a quality rejection threshold before pilot |
| Vocabulary variety | formula correctness; sensitivity to sample length | language and duration | Deterministic output for fixed input; document length sensitivity |
| Response length | unit count accuracy | language | Match manually reviewed reference cases |
| Sentence structure | boundary and connector accuracy | language | Fluent-speaker review with documented disagreements |
| Speech pace | unit/minute error | language, device | Match timestamp reference within a predeclared tolerance |
| Pause pattern | pause count/duration error | device and noise condition | Match annotated timestamps within a predeclared tolerance |
| Repetition | repeated-unit detection | language | Match manually labeled cases and disclose ambiguity |
| Emotional wording | lexicon coverage and false inference risk | language and culture | Describe wording only; do not infer internal emotion |
| Display index | mapping, clipping, and version behavior | feature and language | Raw metric retained; score version stored; no cross-version trend without migration |

Numeric tolerances must be preregistered after a small calibration set and
before evaluation on the held-out validation set. They must not be adjusted
after seeing final results without recording the change as a new protocol.

## Reliability Study

Each participant completes repeat samples under:

- the same device and similar environment;
- a second supported device when available;
- quiet and representative everyday background conditions; and
- more than one speaking prompt to measure topic sensitivity.

Report, as appropriate:

- intraclass correlation or another justified reliability statistic;
- within-person standard deviation;
- confidence intervals;
- minimum detectable change;
- missing and rejected sample rates; and
- results both overall and by supported language.

Until this study is complete, KinaBot must not label small score differences as
meaningful change.

## Multilingual and Fairness Validation

English, Japanese, and Chinese are separate measurement contexts. Validation
must include fluent reviewers and must not assume an English-derived threshold
is universal.

Where sample size and consent permit, evaluate performance by age range,
language, dialect/accent, gender, device class, technical experience,
disability/access need, and country/region. Report sample sizes and uncertainty;
do not publish unstable subgroup rankings based on inadequate data.

For each subgroup, examine transcription failure, rejected recordings,
completion, misunderstanding, score distributions, retention, and reported
harm or burden. A material disparity requires investigation, mitigation, and a
documented release decision.

## Human-Centered Usability Protocol

The study must include older adults, family caregivers, and relevant
professionals. Participants should be asked to explain in their own words:

- what a feature score means and does not mean;
- whether KinaBot assessed health or disease;
- what data is stored and deleted;
- who controls family sharing;
- how to withdraw or delete data; and
- what action they would take after seeing a change.

Core measures include independent task completion, time on task, assistance
required, comprehension, accessibility failures, anxiety or false reassurance,
perceived usefulness, and willingness to continue. Interviewers must record
negative and neutral findings, not only favorable quotations.

## Real-World Pilot Outcomes

The first prospective pilot should designate one primary outcome before
enrollment. Recommended primary outcome:

> The proportion of participants who report that KinaBot helped them prepare a
> clearer, self-directed conversation with a trusted family member, caregiver,
> or professional during the observation period.

Secondary outcomes may include completion, 7/30/90-day retention, voluntary
sharing, non-diagnostic comprehension, perceived autonomy, caregiver burden,
accessibility, privacy requests, and adverse experiences. Engagement alone is
not evidence of health benefit.

## Data and Research Governance

- Product consent is not research consent.
- Research requires a written protocol, appropriate ethics/IRB determination,
  and separate consent where applicable.
- The analysis plan, exclusions, sample size rationale, and primary outcome
  must be dated before final analysis.
- Identity data remains separate from de-identified research data.
- Raw recordings and full transcripts are not retained under the current design.
- Deviations, missing data, withdrawals, and adverse experiences are reported.
- Negative or inconclusive results are preserved.

## Current Evidence and Open Gaps

| Item | Status on 2026-08-05 |
|---|---|
| Automated multilingual and privacy-boundary tests | Available |
| Versioned feature methodology | Available; requires external review |
| Controlled test-retest reliability | Not completed |
| Device and noise robustness | Not completed |
| Fluent-speaker validation dataset | Not completed |
| Older-adult usability study | Not completed |
| Prospective real-world pilot | Not completed |
| Independent replication | Not completed |
| Clinical validity | Not claimed and not established |
| Offline ID/no-email workflow | Automated AppTest passed with synthetic ID `001` |
| Complete disconnected transcription bundle | Pending approved model and wheelhouse acquisition |

## Standards Alignment

The validation program should be reviewed against the WHO ethics and governance
principles for AI in health, NIST AI Risk Management Framework, and relevant
FDA/IMDRF good machine learning practice and transparency principles. Alignment
with a framework is a design discipline, not certification or agency approval.

## Required Artifacts

Every validation activity should preserve:

- dated protocol and version;
- responsible investigator and contributors;
- dataset provenance and consent basis;
- code and scoring version;
- preregistered outcomes and analysis plan;
- complete results, including failures;
- limitations and corrective actions;
- independent review, if any; and
- public summary plus restricted source evidence when privacy requires it.
