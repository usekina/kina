# KinaBot Feature Score Design

## Purpose

KinaBot V1 calculates speech and language feature scores for cognitive wellness reflection.

The goal is to help users observe patterns over time, not to diagnose disease or predict medical risk.

V1 uses multiple feature scores instead of one overall cognitive score.

## Design Principles

Feature scores should be:

- Easy to explain
- Computable from speech or transcript data
- Useful for trend tracking
- Respectful and non-alarming
- Clearly separated from medical diagnosis

## V1 Feature Scores

### 1. Vocabulary Variety

Measures how varied the user's word choices are in a speech sample.

Possible raw metrics:

- Unique words
- Total words
- Unique words / total words

User-facing explanation:

This reflects word variety in this sample. It is not a diagnosis.

### 2. Response Length

Measures how much the user spoke in one test session.

Possible raw metrics:

- Total words
- Total sentences
- Speaking duration
- Words per response

User-facing explanation:

This reflects the amount of speech captured in this session.

### 3. Sentence Complexity

Measures the structure of the user's sentences.

Possible raw metrics:

- Average sentence length
- Number of sentences
- Connecting words
- Clause-like patterns

User-facing explanation:

This reflects sentence structure and expression style in this sample.

### 4. Speech Pace

Measures how quickly the user speaks.

Possible raw metrics:

- Words per minute
- Words per second
- Speaking duration

User-facing explanation:

This reflects speaking pace in this sample. Faster or slower is not automatically better or worse.

### 5. Pause Pattern

Measures pause and silence behavior in the audio sample.

Raw metrics:

- Internal pause ratio within the detected speech span
- Internal pause count, mean, and maximum duration
- Voiced duration and detected speech-span duration
- Leading and trailing recording silence, reported separately and excluded from the score

User-facing explanation:

This reflects pause patterns during speech. In V1, this may be limited by audio processing quality.

### 6. Repetition Pattern

Measures repeated words or repeated phrases.

Possible raw metrics:

- Repeated word ratio
- Repeated phrase count
- Consecutive repeated words

User-facing explanation:

This reflects repetition patterns in this sample. It should be interpreted as a communication feature, not a medical conclusion.

### 7. Emotional Tone

Measures emotional tone from language, and later from audio if supported.

Possible raw metrics:

- Sentiment polarity
- Sentiment subjectivity
- Positive / neutral / negative language tendency

User-facing explanation:

This reflects emotional tone in the language sample. It may be affected by topic, mood, and context.

### 8. Transcription Clarity

Measures whether the speech sample was clear enough for reliable analysis.

Possible raw metrics:

- Transcription success or failure
- Transcript length
- Unrecognized speech warning
- Audio duration versus transcript length

User-facing explanation:

This reflects whether the recording was clear enough for analysis. Low clarity means the user may need to record again.

## Score Range

Each feature may be displayed as a 0-100 feature score.

Scores should be presented as feature levels, not medical grades.

Example:

Vocabulary Variety: 68 / 100

Supporting explanation:

This score reflects word variety in this sample. It is not a diagnosis.

## Raw Metrics And Feature Scores

KinaBot should keep a clear difference between raw metrics and feature scores.

Example:

Raw metric: unique words / total words = 0.62

Feature score: Vocabulary Variety = 68 / 100

This allows the scoring model to improve over time while preserving version history.

## Trend Tracking

Feature scores should be tracked over time.

Trend views may include:

- Current session
- 7-day view
- 30-day view
- 90-day view

If a user tests twice in one day, both sessions should appear as separate data points.

## Responsible Boundary

KinaBot should not display:

- Cognitive age
- Dementia risk
- Medical risk level
- Disease probability
- Diagnosis
- Treatment recommendation
- One overall cognitive score without validation

KinaBot should display:

- Feature-level scores
- Trend changes
- Non-alarming explanations
- Suggestions to consult a qualified professional if users have concerns
