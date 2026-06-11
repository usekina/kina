# KinaBot V1 Implementation Plan

## Goal

Build a fast, simple, low-cost V1 pilot app for up to 100 users over 90 days.

The app should support speech and language-based cognitive wellness reflection while protecting privacy, dignity, and responsible-use boundaries.

## V1 Technical Stack

For local development:

- Python
- Streamlit
- SQLite
- SpeechRecognition or equivalent speech-to-text library
- Local feature scoring functions
- Basic charting with Streamlit or Plotly

For production later:

- Low-cost web hosting or serverless backend
- Managed database
- Email verification service
- Temporary audio handling with automatic deletion
- Usage and cost monitoring

## Development Phases

### Phase 1: App Skeleton

Create a new Aoi-maintained Streamlit app.

Initial files:

- `app.py`
- `database.py`
- `scoring.py`
- `auth.py`
- `config.py`

The first version should run locally before cloud deployment.

### Phase 2: Consent And Access

Build a simple access flow:

1. User enters email.
2. App generates verification code.
3. User enters code.
4. App checks code.
5. User accepts consent notice.
6. User can start a test.

For local development, verification code may be shown on screen instead of sending email.

### Phase 3: Database

Use SQLite for local development.

Tables:

- users
- verification_codes
- test_sessions
- feature_scores
- consent_events

The database should store calculated scores and usage records, not raw audio or full transcripts by default.

### Phase 4: Audio Input

Support one simple input method first:

- Upload audio file

Optional later:

- Browser recording

V1 should process the audio temporarily and delete the file after scoring.

### Phase 5: Speech-To-Text

Use a simple speech-to-text method for V1.

The app should clearly disclose if an external speech-to-text service is used.

Transcripts should not be stored by default.

### Phase 6: Feature Scoring

Implement V1 feature scores:

- Vocabulary Variety
- Response Length
- Sentence Complexity
- Speech Pace
- Pause Pattern
- Repetition Pattern
- Emotional Tone
- Transcription Clarity

Each score should return:

- Raw metrics
- 0-100 feature score
- User-facing explanation
- Scoring model version

### Phase 7: Result View

Show:

- Current session feature scores
- Short respectful explanations
- No medical diagnosis
- No cognitive age
- No dementia risk
- No disease prediction

### Phase 8: Trend View

Show trend charts for:

- 7 days
- 30 days
- 90 days

If a user tests twice in one day, show both sessions as separate points.

### Phase 9: Data Deletion

Add a simple deletion request process.

At minimum, document how a user can request deletion.

Later versions may add self-service deletion.

### Phase 10: Admin Metrics

Track basic pilot metrics:

- Registered users
- Active users
- Total tests
- Tests per day
- Average tests per user
- Error count
- Estimated cost per test

Admin metrics should not expose raw audio or transcripts.

## Not In V1

V1 will not include:

- Mobile app
- Payment system
- Insurance workflow
- Medical diagnosis
- Cognitive age
- Dementia risk
- Clinical decision support
- Family sharing
- Long-term raw audio storage

## V1 Completion Criteria

V1 is complete when:

- A user can verify access
- A user can accept consent
- A user can upload audio
- The app can generate feature scores
- The app saves scores without saving raw audio or transcript
- The app shows current results
- The app shows basic trend history
- The app limits users to 2 tests per day
- The app can run locally
