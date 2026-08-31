# Institutional Pilot Evidence Framework

This directory defines how KinaBot documents university, research, nonprofit,
and community pilots without overstating adoption, endorsement, scientific
validity, or health impact.

An institution must not be described publicly as a KinaBot partner, customer,
deployment, or endorser merely because someone affiliated with it viewed the
project, exchanged email, attended a demonstration, or tested the software
informally. Public institutional claims require written authorization and
evidence appropriate to the claim.

## Status Vocabulary

Use exactly one of these statuses in an approved public pilot summary:

| Status | Minimum evidence |
|---|---|
| Exploratory contact | Private dated correspondence; do not name the institution publicly without permission |
| Pilot approved | Written scope, named accountable contacts, authorization, and applicable privacy or ethics review |
| Pilot active | Approved pilot plus a dated deployment or participation record and the exact software release used |
| Pilot completed | Active-pilot evidence plus documented completion, limitations, and outcome calculations |
| Findings published | Completed pilot plus an authorized public report, publication, or independently accessible record |

None of these statuses implies institutional endorsement. Logos, trademarks,
quotes, and individual names require explicit permission.

## Required Evidence

Before publishing a pilot summary, preserve the following in an access-
controlled evidence file:

1. written scope, dates, roles, and institutional contact;
2. the product version, scoring version, deployment mode, and change log;
3. approval or documented determination for privacy, security, consent, and
   research-ethics requirements;
4. participant eligibility, exclusions, withdrawals, and missing-data rules;
5. metric definitions, source records, numerator, denominator, and calculation;
6. adverse events, complaints, negative results, limitations, and corrections;
7. authorization for every public institution name, quotation, or logo; and
8. an independent contact who can verify the institution's actual involvement.

Raw voice recordings, transcripts, personal data, agreements, private contact
information, credentials, and confidential university material must never be
committed to this public repository.

## Public Evidence Layout

Create a directory only after public disclosure is authorized:

```text
institutional-pilots/
  YYYY-MM-approved-public-name/
    PUBLIC-SUMMARY.md
    RELEASE-AND-METHOD.md
    AGGREGATE-RESULTS.md
    LIMITATIONS-AND-CORRECTIONS.md
```

Start from [PILOT-TEMPLATE.md](PILOT-TEMPLATE.md). If a field is unknown, report
it as unknown. Do not replace missing evidence with promotional language.

## Claims Boundary

KinaBot is not a medical device or diagnostic system. A pilot may evaluate
usability, accessibility, speech-feature reliability, privacy comprehension,
or communication/reflection outcomes. It must not be presented as proving
dementia detection, disease prediction, treatment benefit, or clinical utility
without an appropriate validated study and authorization.

The governing measurement and publication rules are in
[IMPACT-METRICS.md](../IMPACT-METRICS.md),
[VALIDATION-PLAN.md](../VALIDATION-PLAN.md), and
[SCIENTIFIC-EVIDENCE-AND-CLAIMS.md](../SCIENTIFIC-EVIDENCE-AND-CLAIMS.md).
