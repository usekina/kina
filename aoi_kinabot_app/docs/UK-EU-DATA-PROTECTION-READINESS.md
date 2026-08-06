# UK and EU University Research Data-Protection Readiness

Status: engineering readiness checklist, not legal certification

KinaBot Offline Research Mode is designed to support a university's UK GDPR,
EU GDPR, ethics, and information-security review. The university remains the
data controller unless its written arrangement determines otherwise. AImoji LLC
and any participating researcher must document their actual roles before data
collection.

## Implemented Privacy-by-Design Controls

| Principle | Offline-mode control |
|---|---|
| Purpose limitation | Mode is limited to approved speech-reflection research; no diagnostic or employment use |
| Data minimisation | No email is required; school ID is hashed; raw audio and full transcript are not retained |
| Pseudonymisation | A study-secret HMAC key links sessions without storing the assigned ID |
| Local control | SQLite, transcription, scoring, and exports remain on the approved research computer |
| External transfer reduction | SMTP and OpenAI are disabled; local model path is mandatory |
| Transparency | UI states offline mode and non-medical boundaries; study notice remains the university's responsibility |
| Accuracy and interpretation | Versioned descriptive features; no diagnosis or disease-risk claim |
| Security | Local-only default bind, explicit model path, temporary-file deletion, and separate re-identification key |
| Accountability | Git version, scoring version, validation log, risk register, and study acceptance test |

Pseudonymised data remains personal data when the university can reconnect it
to a participant. It must not be described as anonymous solely because KinaBot
does not store the original ID.

## Required Before a UK University Study

The university's principal investigator, ethics body, Information Governance
team, and Data Protection Officer should determine and document:

- research purpose and protocol;
- controller, joint-controller, and processor roles;
- Article 6 lawful basis;
- Article 9 condition if health or other special-category data is processed;
- whether a Data Protection Impact Assessment is required;
- participant information and research-consent process;
- whether participation consent differs from the data-processing lawful basis;
- retention, deletion, withdrawal, access, and correction procedures;
- device encryption, physical security, backups, and incident response;
- who holds the participant-ID linkage key;
- whether any data leaves the UK/EEA and the applicable transfer mechanism;
- minimum cell sizes and disclosure controls for publications; and
- prohibition on decisions or measures about individual participants based on
  exploratory KinaBot results.

## Recommended Institutional Configuration

- Generate non-meaningful IDs such as `001`, `002`, and `003`.
- Store the linkage file in a separate approved system with narrower access.
- Keep KinaBot's database on an encrypted, managed device.
- Use a dedicated OS account without general browsing or email.
- Disable network adapters during collection if the protocol requires air-gap
  operation.
- Restrict research export access to named study personnel.
- Record every export and deletion.
- Destroy the linkage key and/or local dataset according to the approved
  retention schedule.

## Official Review References

- UK Information Commissioner's Office, *The research provisions*:
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/
- ICO, *Principles and grounds for processing*:
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/principles-and-grounds-for-processing/
- ICO, *Special category data*:
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/special-category-data/what-are-the-rules-on-special-category-data/

ICO guidance is evolving following the UK Data (Use and Access) Act 2025. The
university should use the current guidance at the time of protocol approval.
