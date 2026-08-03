# Mobile-First Results Architecture

## Goal

An ordinary user should record or upload a sample, understand the immediate
result, and revisit personal trends from a phone without navigating a long
research-style page.

## Participant Information Architecture

```text
Sign in
├── Today
│   ├── Add voice sample
│   ├── Analyze
│   ├── Four-dimension reflection
│   └── Compact eight-feature snapshot
└── Trends
    ├── Latest eight-feature snapshot
    ├── Feature selector
    ├── Recent 3 / All sessions
    ├── Descriptive change statement
    └── Scoring methodology
```

The top-level **Today / Trends** control makes the latest result available in
one tap after sign-in. The eight-feature snapshot uses a two-column grid on
mobile. Trend charts show one selected feature at a time to reduce clutter and
avoid horizontal scrolling.

## Accessibility Principles

- No laptop is required for the participant journey.
- Core controls remain large enough for touch input.
- Upload remains the default reliable audio path; direct recording is optional.
- No result depends on color alone.
- Detail is progressively disclosed without hiding consent or limitations.
- The application avoids labels such as normal, abnormal, improved, declined,
  or at risk.

## Research Administration

Research Admin is owner-only and must remain separated from the participant
journey. A future dedicated admin route should use a responsive drawer or
bottom sheet on small screens and preserve separation between private identity
exports and de-identified research records.

## Evaluation

Measure task completion, time to find the latest result, taps to reach Trends,
scroll depth, abandonment, comprehension of the 0–100 scale, and usability by
older adults. Record only verified results in `IMPACT.md`.

