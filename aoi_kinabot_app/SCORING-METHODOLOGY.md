# KinaBot V1.1 Scoring Methodology

Scoring model version: `score-v4-internal-pause-span`

## Purpose

KinaBot produces repeatable, descriptive speech and language features for one
voice sample and, after multiple sessions, shows changes relative to the same
user's earlier samples.

The scores are not medical measurements. They are not validated measures of
intelligence, cognitive status, cognitive decline, disease, or risk.

## Processing Boundary

1. A user selects an audio file from a phone or computer.
2. KinaBot creates a temporary working file.
3. A private/local Whisper model produces a transcript and timestamps.
4. KinaBot applies a language-specific NLP adapter.
5. Python calculates raw metrics and 0–100 display scores.
6. KinaBot stores the raw metrics, display scores, language, and scoring
   version.
7. The temporary audio and full transcript are discarded.

OpenAI is not involved in transcription or feature scoring.

## Display Scale

KinaBot displays every feature as a **0–100 sample feature score**, for
example `56 / 100`.

- It is an engineered display index, not a percentage or probability.
- It is not a percentile or comparison with other users.
- It is not a health, ability, diagnostic, or risk score.
- A population-standardized score such as a T-score will not be used unless
  KinaBot completes an appropriate reference-sample calibration and validation
  study.

Feature names, explanations, and result boundaries are shown in the language
selected for the recording. Explanations state the observable feature directly
and avoid repetitive introductory wording.

## Language Adapters

| Language | Segmentation | Language-specific inputs |
|---|---|---|
| English | Regular-expression word tokens | English connectors and emotional-expression vocabulary |
| Japanese | Janome morphological analysis | Japanese connectors and emotional-expression vocabulary |
| Chinese | jieba segmentation | Chinese connectors and emotional-expression vocabulary |

Length and pace normalization use language-specific V1 reference centers.
These engineering centers require further calibration with consented,
representative recordings before broad interpretation.

## Feature Set

| Feature | Primary raw evidence |
|---|---|
| Vocabulary Variety | Unique units, total units, type-token ratio |
| Response Length | Total language units relative to the language adapter |
| Sentence Complexity | Units per sentence and discourse connectors |
| Speech Pace | Language units per minute |
| Pause Pattern | Voiced time, internal pause time/count, mean/maximum pause, internal pause ratio |
| Repetition Pattern | Repeated units relative to total units |
| Emotional Tone | Counts from a small language-specific expression lexicon |
| Transcription Clarity | Amount of recognizable transcript evidence |

English discourse connectors are counted only when their normalized tokens
match complete connector token sequences. Substrings inside unrelated words do
not count; for example, `and` in `candy` and `if` in `gift` are excluded. Each
real occurrence is counted, including repeated connectors. Japanese and Chinese
use their language-specific connector matching behavior rather than English
word-boundary rules.

Pause Pattern measures gaps between detected speech segments within the detected
speech span. The internal pause ratio is total positive gap time divided by the
interval from the first valid speech start to the last valid speech end. Leading
and trailing recording silence are retained as separate raw metrics but do not
affect the Pause Pattern score. Overlapping segments are merged before voiced
and pause durations are calculated.

## Longitudinal Display

- Sessions 1–2: display the current sample only.
- Session 3 onward: display descriptive history and first-to-latest
  differences.
- A difference is not automatically a meaningful personal change.
- A lower score is not automatically a worse health state.
- Changes may reflect topic, language choice, microphone, environment,
  recording length, fatigue, mood, transcription error, or ordinary variation.

## Versioning

Every stored session includes:

- application version;
- consent version;
- scoring-model version;
- language; and
- session metadata.

Any future change to tokenization, reference centers, feature formulas, or
feature definitions must increment the scoring-model version. Scores from
different versions should not be placed on one uninterrupted trend line
without a documented migration or recalculation method.

The application therefore limits displayed trend calculations to sessions from
the current scoring-model version. Older sessions remain available in research
exports with their original version metadata.

## Validation Status

V1.1 is an engineering and usability pilot. Automated tests verify code paths,
ranges, language selection, data minimization, and timestamp-based pause
calculation. They do not establish clinical validity.

Future validation should include:

- fluent-speaker review for every language;
- repeat recordings under controlled and everyday conditions;
- transcription error analysis;
- device and microphone robustness testing;
- test-retest reliability;
- subgroup fairness review; and
- documented calibration changes.

Whisper segment timestamps cannot detect every pause that occurs inside one
segment. More precise word timestamps, audio-level voice activity detection,
and real-recording calibration remain future validation work.
