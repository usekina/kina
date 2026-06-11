# KinaBot Data And Access Design

## Purpose

This document defines the V1 data access, verification, storage, and retention design for the Aoi-maintained KinaBot app.

The goal is to support a small public-benefit pilot while protecting user dignity, privacy, and responsible-use boundaries.

## V1 Access Model

KinaBot V1 will use email-based access control.

User flow:

1. User enters an email address.
2. System sends a one-time verification code.
3. User enters the code.
4. If the code is valid, the user may start a test.
5. Each verified user may complete up to 2 tests per day.

The access model is designed to:

- Prevent uncontrolled public usage
- Limit server and AI/transcription cost
- Count real users and active usage
- Support responsible pilot management
- Reduce spam and abuse

## Email Handling

The system should avoid storing plain email addresses when possible.

Preferred storage:

- Hashed email
- Verification code hash
- Code expiration time
- Verification status
- Consent version
- User creation time
- Last active time

Plain email may be temporarily used only for sending the verification code.

## Test Limit Rules

Each verified user may complete:

- Maximum 2 tests per calendar day
- Maximum 1 active test session at a time

Each test should create one test record.

If a user completes two tests on the same day, the records should be labeled:

- Session 1
- Session 2

Trend charts should use exact timestamps, not only daily averages.

Daily averages may be calculated later for summary views.

## Data Stored Per Test

Each test record should store:

- Anonymous user ID or hashed email ID
- Test ID
- Timestamp
- Session number for the day
- App version
- Consent version
- Feature score version
- Calculated feature scores
- Processing status
- Error status if applicable

Example feature scores may include:

- Vocabulary variety
- Sentence complexity
- Speech pace
- Response length
- Repetition pattern
- Emotional tone
- Pause or silence pattern
- Transcription clarity

## Data Not Stored By Default

KinaBot V1 should not store by default:

- Raw audio files
- Full transcripts
- Medical history
- Family member information
- Personal diary content
- Free-text health descriptions

Raw audio should be processed temporarily and deleted after feature extraction unless the user explicitly gives separate consent for research storage.

Transcripts should not be saved by default. If transcripts are needed for debugging or research, separate consent should be required.

## Trend View Design

The system should support trend views over:

- Current session
- 7 days
- 30 days
- 90 days

Each feature score can be shown as a line chart over time.

For users with two tests in one day:

- Both points should be shown on the chart
- Session 1 and Session 2 should be distinguishable
- A daily average may be shown later as an optional summary

## User Deletion

Users should be able to request deletion of their records.

Deletion should remove or anonymize:

- User access record
- Verification history
- Test records
- Feature score history connected to that user

Aggregate anonymous usage statistics may be retained if they cannot reasonably identify a user.

## Admin Metrics

The admin system may track:

- Total registered users
- Active users
- Tests completed
- Tests per user
- Daily and monthly usage
- Average system cost per test
- Error rate
- Deletion requests
- Consent version adoption

Admin metrics should not expose raw audio or transcripts.

## Future Architecture Direction

For local development:

- Streamlit or simple web app
- Local database such as SQLite
- Local email testing or manual verification code

For production:

- Web frontend
- Serverless backend
- Database for user and score records
- Email service for verification codes
- Temporary file handling with automatic deletion
- Monitoring and cost controls

Potential production stack:

- Frontend: simple web app or hosted Streamlit alternative
- Backend: AWS Lambda or similar serverless function
- Database: DynamoDB or managed Postgres
- Email: Amazon SES or equivalent service
- Temporary storage: S3 with lifecycle deletion if needed

## Responsible Boundary

KinaBot V1 stores feature scores for cognitive wellness reflection and trend awareness.

The system should not store or display outputs as medical diagnosis, dementia risk, cognitive age, disease prediction, or treatment recommendation.
