# KINA Cognitive Health Signals Documentation

## Overview

KINA (Kognitive Intelligence Neural Assessment) analyzes speech patterns to detect cognitive changes through four key biomarkers. This document explains how each signal is calculated, weighted, and interpreted to provide a comprehensive cognitive health assessment.

## Signal Architecture

The KINA system processes audio recordings through multiple analysis layers to extract cognitive health indicators:

```
Audio Input → Speech Recognition → Text Analysis → Signal Extraction → Cognitive Scoring
```

## Core Signals

### 1. Lexical Diversity (30% Weight)

**Purpose**: Measures vocabulary richness and word variety in speech.

**Calculation**:
```python
lexical_diversity_score = unique_words / total_words
```

**Scoring Ranges**:
- **Optimal (100 points)**: ≥ 0.75 (75%+ unique words)
- **Good (60-100 points)**: 0.6-0.75 (60-75% unique words)
- **Fair (30-60 points)**: 0.4-0.6 (40-60% unique words)
- **Poor (0-30 points)**: < 0.4 (< 40% unique words)

**Clinical Significance**:
- High diversity indicates rich vocabulary and cognitive flexibility
- Low diversity may suggest word-finding difficulties or cognitive decline
- Healthy adults typically maintain 60-90% lexical diversity

**Example**:
```
Text: "The beautiful garden has beautiful flowers and beautiful trees"
Total words: 9
Unique words: 6 (the, beautiful, garden, has, flowers, and, trees)
Diversity score: 6/9 = 0.67 (Good range)
```

### 2. Speech Fluency (25% Weight)

**Purpose**: Evaluates speaking rate and temporal aspects of speech production.

**Calculation**:
```python
words_per_second = total_words / recording_duration_seconds
```

**Scoring Ranges**:
- **Optimal (100 points)**: 2.0-3.0 words/second
- **Good (80 points)**: 1.5-2.0 or 3.0-3.5 words/second
- **Fair (60 points)**: 1.0-1.5 or 3.5-4.0 words/second
- **Poor (40 points)**: 0.5-1.0 or 4.0-5.0 words/second
- **Very Poor (20 points)**: < 0.5 or > 5.0 words/second

**Clinical Significance**:
- Normal speaking rate indicates healthy motor speech control
- Too slow may suggest processing difficulties or motor impairment
- Too fast may indicate anxiety, mania, or loss of speech control
- Optimal range reflects natural, comfortable speech patterns

**Example**:
```
Recording: 30 seconds
Total words: 75
Speaking rate: 75/30 = 2.5 words/second (Optimal range)
```

### 3. Sentence Complexity (25% Weight)

**Purpose**: Assesses syntactic complexity and grammatical sophistication.

**Calculation**:
```python
avg_sentence_length = total_words / number_of_sentences
conjunction_count = count_of_connecting_words
```

**Tracked Conjunctions**: 'and', 'but', 'or', 'because', 'although', 'since', 'while', 'if', 'when', 'though'

**Scoring Ranges**:
- **Optimal (100 points)**: 12-20 words/sentence + ≥1 conjunction
- **Good (80 points)**: 8-12 or 20-25 words/sentence
- **Fair (60 points)**: 6-8 or 25-30 words/sentence
- **Poor (40 points)**: 4-6 or >30 words/sentence
- **Very Poor (20 points)**: <4 words/sentence

**Clinical Significance**:
- Complex sentences indicate preserved executive function
- Simple sentences may suggest cognitive simplification
- Appropriate conjunction use shows logical thinking
- Extremely long sentences may indicate disorganized thinking

**Example**:
```
Text: "I went to the store because I needed groceries, and I also wanted to buy flowers."
Sentences: 1
Words: 17
Conjunctions: 2 ('because', 'and')
Average length: 17 words/sentence (Optimal range)
```

### 4. Emotional Expression (20% Weight)

**Purpose**: Evaluates emotional tone and affective language patterns.

**Calculation**:
```python
polarity_score = sentiment_analysis(text)  # Range: -1 to +1
```

**Scoring Ranges**:
- **Optimal (100 points)**: -0.1 to +0.3 (slightly positive to neutral)
- **Good (80 points)**: -0.3 to -0.1 or +0.3 to +0.5
- **Fair (60 points)**: -0.5 to -0.3 or +0.5 to +0.7
- **Poor (40 points)**: < -0.5 or > +0.7

**Clinical Significance**:
- Balanced emotional expression indicates healthy affect regulation
- Extreme negativity may suggest depression or anxiety
- Extreme positivity may indicate mania or inappropriate affect
- Neutral to slightly positive is considered healthiest

**Example**:
```
Text: "I feel pretty good today and enjoyed my morning walk"
Polarity: +0.2 (slightly positive)
Score: 100 points (Optimal range)
```

## Overall Cognitive Score Calculation

The final cognitive health score is calculated using weighted components:

```python
overall_score = (
    lexical_diversity_component * 0.30 +
    speech_fluency_component * 0.25 +
    sentence_complexity_component * 0.25 +
    emotional_expression_component * 0.20
)
```

### Risk Level Classification

| Score Range | Risk Level | Color | Description |
|-------------|------------|-------|-------------|
| 80-100 | Low Risk | 🟢 | Excellent cognitive health indicators |
| 65-79 | Low-Moderate Risk | 🟡 | Good cognitive health with minor areas for attention |
| 50-64 | Moderate Risk | 🟠 | Some cognitive health concerns detected |
| 0-49 | Higher Risk | 🔴 | Multiple cognitive health indicators suggest consultation recommended |

## Cognitive Age Estimation

KINA estimates cognitive age using a simplified model:

```python
score_deviation = 85 - overall_score  # 85 is considered optimal baseline
age_adjustment = score_deviation * 0.3  # Each point below 85 adds ~0.3 years
baseline_age = 35  # Healthy adult baseline
estimated_cognitive_age = max(20, baseline_age + age_adjustment)
```

**Interpretation**:
- Cognitive age reflects functional cognitive capacity
- Lower than chronological age suggests good cognitive health
- Higher than chronological age may indicate cognitive concerns
- Minimum age is capped at 20 years for practical interpretation

## Personalized Recommendations

The system generates targeted recommendations based on component scores:

### Lexical Diversity < 70
- 📚 "Expand vocabulary through reading diverse materials"

### Speech Fluency < 70
- **If too slow** (< 1.5 words/sec): 🗣️ "Practice speaking exercises to improve fluency"
- **If too fast** (> 1.5 words/sec): ⏸️ "Practice paced speaking to improve clarity"

### Sentence Complexity < 70
- 🧠 "Practice using varied sentence structures and connecting words"

### Emotional Expression < 70
- 😊 "Consider activities that promote emotional well-being"

### All Components ≥ 70
- 🎉 "Excellent results! Continue maintaining cognitive health through regular mental activities"

## Technical Implementation

### Signal Processing Pipeline

1. **Audio Preprocessing**:
   - Convert to 16kHz WAV format
   - Noise reduction and normalization

2. **Speech Recognition**:
   - Google Speech Recognition API
   - Multi-language support (English, Japanese)

3. **Text Analysis**:
   - Tokenization and sentence segmentation
   - Part-of-speech tagging
   - Sentiment analysis using TextBlob

4. **Feature Extraction**:
   - Word counting and uniqueness calculation
   - Temporal analysis (duration, speaking rate)
   - Syntactic complexity measurement
   - Emotional polarity assessment

5. **Scoring and Interpretation**:
   - Component score calculation
   - Weighted aggregation
   - Risk classification
   - Recommendation generation

### Data Flow

```
Raw Audio → Preprocessed Audio → Transcribed Text → Linguistic Features → Cognitive Scores → Clinical Interpretation
```

## Validation and Reliability

### Signal Validation
- Each signal is based on established psycholinguistic research
- Scoring ranges derived from normative data studies
- Component weights optimized for cognitive health assessment

### Quality Assurance
- Minimum recording duration: 10 seconds
- Maximum recording duration: 60 seconds
- Audio quality validation
- Transcription confidence scoring

### Limitations
- Requires clear audio quality
- Language-dependent (currently English/Japanese)
- Not a replacement for clinical assessment
- Influenced by education level and cultural factors

## Clinical Applications

### Screening Tool
- Early detection of cognitive changes
- Monitoring cognitive health over time
- Complement to traditional assessments

### Research Applications
- Longitudinal cognitive studies
- Treatment efficacy monitoring
- Population health screening

### Healthcare Integration
- Telemedicine cognitive assessments
- Remote patient monitoring
- Clinical decision support

## Future Enhancements

### Planned Signal Additions
- **Semantic Fluency**: Category-based word generation
- **Phonemic Complexity**: Sound pattern analysis
- **Discourse Coherence**: Topic maintenance and transitions
- **Prosodic Features**: Rhythm, stress, and intonation patterns

### Technical Improvements
- Real-time processing capabilities
- Enhanced multi-language support
- Machine learning model optimization
- Integration with wearable devices

## References and Research Basis

The KINA signal calculations are based on established research in:
- Computational linguistics and natural language processing
- Neuropsychological assessment methodologies
- Speech pathology and cognitive neuroscience
- Digital biomarker development for cognitive health

---

*This documentation is part of the KINA (Kognitive Intelligence Neural Assessment) system. For technical support or clinical questions, please refer to the main documentation or contact the development team.*