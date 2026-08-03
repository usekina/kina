# Mobile Results, Navigation, and Scoring Transparency

Date: 2026-08-02  
Source: Anonymized external product tester  
Context: Smartphone in portrait orientation; completed the single-session
analysis flow  
Consent: Feedback is paraphrased for public documentation  
Status: Accepted and implemented for validation

## Observations

- The single-session analysis flow felt smooth.
- A user may want to understand how the eight displayed features are derived.
- A returning user may want to revisit at least the latest eight scores.
- The existing portrait layout required excessive vertical scrolling.
- A compact panel or navigation structure reachable within two taps may be more
  usable than one long page.
- Research administration needs a mobile-appropriate presentation, but likely
  participants, caregivers, and older adults should not need a laptop to use
  the participant experience.

## Product Interpretation

The feedback identifies four distinct needs:

1. scoring transparency;
2. quick access to the latest result;
3. mobile information density; and
4. separation between participant and research-administration workflows.

## Design Response

- Add a top-level **Today / Trends** control after sign-in.
- Show the latest eight features in a compact two-column mobile grid.
- Let the user select one feature at a time for a readable trend chart.
- Offer recent-three-session and all-session views.
- Put concise feature definitions in the interface and link to the versioned
  scoring methodology.
- Keep Research Admin permission-gated and outside the participant's primary
  task flow. A dedicated admin route or mobile drawer remains future work.

## Responsible Boundary

Scores are engineered descriptive indexes for a sample. A trend is not a
diagnosis, cognitive assessment, or proof of improvement or decline. Topic,
language, microphone, environment, recording length, fatigue, mood, and
transcription quality may affect a result.

## Validation Needed

- Can a new mobile user find Trends without instruction?
- Can a returning user locate the latest eight features within two taps?
- Does the two-column grid remain readable at 320–430 pixel widths?
- Do users understand the difference between a feature index and a health
  score?
- Does selecting one metric reduce chart confusion?
- Do older adults complete the flow without a laptop or caregiver assistance?

## Evidence and Links

- Implementation issue: https://github.com/usekina/kina/issues/26
- Pull request: https://github.com/usekina/kina/pull/27
- Release and measured outcomes: pending usability testing
