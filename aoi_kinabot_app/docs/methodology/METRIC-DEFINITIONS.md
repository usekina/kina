# How KinaBot Produces the Eight Feature Indexes

Model version: `score-v2-multilingual`

KinaBot uses its own Python and multilingual NLP pipeline. Audio is transcribed
privately by the application, then language-specific tokenization and acoustic
timing evidence are used to calculate raw metrics. OpenAI does not transcribe
the recording or calculate these scores.

Each displayed value is a bounded **0–100 sample feature index**. It is not a
percentage, probability, percentile, population ranking, health score, or
cognitive test result.

| Feature | Raw evidence | Display transformation |
|---|---|---|
| Vocabulary Variety | Distinct language units divided by total units | Type-token ratio mapped to 0–100 |
| Response Length | Total recognized language units | Units relative to a language-specific pilot reference center |
| Sentence Structure | Average units per sentence and connector count | Weighted, bounded combination of length and connectors |
| Speech Pace | Language units and recording duration | Distance from a broad language-specific pilot center |
| Pause Pattern | Pause ratio, count, mean, maximum, voiced and pause time | Weighted distance from broad descriptive centers |
| Repetition Pattern | Repeated instances relative to total units | Lower repetition ratio maps to a higher display index |
| Emotional Tone | Small language-specific positive/negative wording lexicon | Bounded balance of detected wording; topic strongly affects it |
| Recording Clarity | Amount of recognizable transcript evidence | Tiered index based on usable recognized units |

English uses regular-expression word tokens, Japanese uses Janome
morphological analysis, and Chinese uses jieba segmentation. Pace and response
length use language-specific pilot centers. These centers are engineering
choices that require calibration and validation with representative consented
data.

## Longitudinal Interpretation

KinaBot compares a person only with that person's earlier samples. A higher or
lower value is not inherently healthier or worse. Small differences may be
ordinary variation, and even sustained differences require careful research
before any relationship with cognitive wellness can be claimed.

Formula-level implementation is versioned in `scoring.py`; the broader
methodology, limitations, and validation plan are maintained in
`SCORING-METHODOLOGY.md`.

