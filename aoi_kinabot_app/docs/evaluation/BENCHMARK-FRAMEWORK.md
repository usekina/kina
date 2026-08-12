# KinaBot Benchmark and Evaluation Framework

**Status:** Public evaluation specification; not a report of completed results
**Purpose:** Make technical and human-centered evaluation reproducible

KinaBot should not be benchmarked by one accuracy number. A longitudinal,
multilingual system must be evaluated across transcription, feature
reproducibility, reliability, robustness, human understanding, privacy, and
operations.

## Benchmark layers

| Layer | Question | Candidate measures |
|---|---|---|
| Transcription | Does speech-to-text perform consistently across languages and conditions? | WER/CER, failure rate, latency, language-stratified error |
| Feature engine | Are derived features deterministic and versioned? | Same-input consistency, regression fixtures, provenance completeness |
| Reliability | Are measurements stable under comparable repeated conditions? | Test-retest estimates, within-person variance, minimum detectable change after study design |
| Robustness | How do device, duration, noise, and task changes affect output? | Stratified error, sensitivity analysis, audio-quality eligibility |
| Human experience | Do people understand the result and its limitations? | Completion, comprehension, misinterpretation, accessibility, deletion-task success |
| Privacy and governance | Does implementation match the stated data lifecycle? | Audio deletion, transcript non-retention, authorization, pseudonymization, deletion tests |
| Operations | Can the service remain reliable and sustainable? | Availability, error rate, latency, cost per successful analysis, retry correctness |
| Claims compliance | Does every surface preserve the non-diagnostic boundary? | Structured content review and prohibited-claim tests |

## Minimum benchmark protocol

Every evaluation should publish or retain:

1. research question and intended use;
2. software, scoring, transcription, and adapter versions;
3. language, device, task, recording condition, and inclusion criteria;
4. dataset source, consent, governance, and representativeness limits;
5. metrics selected before viewing favorable results;
6. uncertainty, missingness, exclusions, and negative findings;
7. reproducible test code or a documented reason it cannot be shared;
8. allowed conclusion and prohibited conclusion;
9. independent reviewer or replication status.

## Comparison rules

Compare KinaBot with a category or named system only when task, population,
language, input quality, intended use, and metric are aligned. Do not claim
superiority from feature lists or marketing.

| Category | Appropriate comparison question |
|---|---|
| Voice journal | Does it support user-controlled recording and meaningful longitudinal review? |
| Consumer wellness product | Are trends, privacy controls, and claims understandable? |
| Research speech pipeline | Are methods, versions, handling, and outputs reproducible? |
| Clinically validated assessment | What external validation and intended use exist that KinaBot does not yet have? |

The current system supports descriptive longitudinal reflection. Clinical
validity remains an evidence gap, not a benchmark score to infer.

## Student benchmark projects

- **Multilingual reliability:** stratify transcription and feature stability by
  language without assuming cross-language score equivalence.
- **Recording robustness:** test duration, device, distance, and controlled
  noise using consented or synthetic fixtures.
- **Dignity-first comprehension:** test whether users understand observable
  variation and do not infer diagnosis.
- **Private API verification:** test authorization, HMAC pseudonyms, retention,
  deletion, provenance, and localhost-only exposure.
- **Responsible usage policy:** verify that failures do not consume quota and
  study settings do not silently change public policy.

Never publish participant audio, transcripts, raw identifiers, secrets, or
small-cell results that create re-identification risk. Engineering tests do not
establish health benefit, clinical validity, legal compliance, or regulatory
authorization.
