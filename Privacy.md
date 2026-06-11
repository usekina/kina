# Privacy

KinaBot works with sensitive voice, speech, and language data. Privacy and consent are core requirements for any responsible use of this project.

## Consent First

Only record, upload, or analyze speech when you have valid consent from every person whose voice or personal information may be included.

This is especially important when using KinaBot with older adults, family members, caregivers, or anyone who may not fully understand how audio data will be processed.

## What Data May Be Processed

Depending on how KinaBot is deployed, the system may process:

- Audio recordings
- Speech transcripts
- Language pattern metrics
- Generated reports
- Basic usage or error information

Do not upload highly sensitive information unless the deployment environment, consent process, and data handling rules are appropriate for that use.

## Current Prototype Data Flow

The current Python prototype may:

- Record audio through the browser interface
- Save recordings locally during analysis
- Convert speech to text using the `SpeechRecognition` library
- Use Google speech recognition services unless replaced by a local or private transcription backend
- Generate reports that may include transcripts and speech pattern summaries

Users and deployers should review the code and deployment settings before using KinaBot with real personal data.

## Data Retention

The prototype may save recordings in a local `recordings` folder. Before production use, deployers should define:

- Whether recordings are stored
- How long recordings are kept
- Who can access recordings and reports
- How users can request deletion
- Whether analysis can be performed without retaining raw audio

For privacy-sensitive deployments, KinaBot should minimize retention and prefer deleting raw audio after analysis unless the user explicitly chooses to save it.

## External Services

Some speech recognition workflows may send audio or derived data to external services. If external services are used, users should be clearly informed before recording or uploading audio.

For high-privacy use cases, consider replacing cloud transcription with a local or enterprise-approved speech-to-text system.

## User Responsibility

Users are responsible for following applicable privacy, recording consent, data protection, and healthcare-related laws in their jurisdiction.

KinaBot should not be used for hidden recording, surveillance, coercive monitoring, or any use that reduces the dignity or autonomy of older adults.

