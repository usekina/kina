# Changelog

## 2026-08-06 · Dated learning case library

- Established a publication standard and reusable template for meaningful,
  evidence-linked product learning cases.
- Published the first case on redesigning an online product for offline
  university research, including alternatives, failures, corrections, tests,
  social context, student work, unresolved risks, and claims boundaries.
- Linked the case library to feedback, engineering learning, the open knowledge
  center, and planned data-science teaching material.
- Expanded the library into a product learning archive with a complete journey
  timeline, commit/PR evidence map, globally reusable open curriculum, and
  September 2026 white paper preparation roadmap.
- Separated collaborative exploration from the independently developed
  Aoi-maintained application so contributor credit and current product
  responsibility remain visible.

## 2026-08-05 · Offline university research mode

- Added school-assigned participant IDs without email collection or SMTP.
- Added study-secret HMAC pseudonyms so low-entropy IDs such as `001` are not
  stored or represented by enumerable unkeyed hashes.
- Enforced local Whisper model paths and disabled OpenAI in offline mode.
- Added offline-only dependencies, installation, launch, bundle-build, checksum,
  verification, quick-start, and acceptance-test materials.
- Added UK/EU data-protection readiness and a reusable learning record without
  claiming institutional approval or legal compliance.

## 2026-08-03 · 30-day pattern experience

- Added the multilingual **30 Days to Know Your Patterns** experience.
- Reframed daily use around one recommended 60-second reflection; second and
  third check-ins remain optional.
- Added calendar progress, completed reflection-day count, and a no-streak,
  no-penalty design for missed days.
- Kept trends available after three total sessions rather than requiring three
  sessions in one day.

## 2026-08-02 · Mobile results and open learning record

- Added a compact two-column eight-feature result grid for narrow screens.
- Added top-level Today and Trends navigation after sign-in.
- Added latest-result recall and one-feature-at-a-time recent/all trend views.
- Added in-app scoring transparency linked to the versioned methodology.
- Published anonymized feedback, mobile UX, metric-definition, and university
  teaching-case documentation.

## 2026-07-31 · Multilingual landing experience

- Added a warm, minimal first screen built around the brand line
  "Your Voice, Your Patterns, Over Time"
- Added English, Japanese, and Chinese interface selection before sign-in.
- Localized the first-screen sign-in and verification flow and uses the
  selected interface language as the initial recording language.

## 2026-07-31

- Temporarily disabled in-browser recording after unreliable mobile-browser
  behavior was observed. The interface now clearly marks direct recording as
  coming soon and uses the stable audio-file upload flow.

All notable changes to the current Aoi-maintained KinaBot application are
recorded here. Historical repository-root prototypes are outside this
changelog.

## [1.1.0] - Unreleased

### Added

- English, Japanese, and Chinese language selection.
- Language-specific tokenization with English rules, Janome for Japanese, and
  jieba for Chinese.
- Versioned, local Python/NLP feature scoring.
- Private local transcription with `faster-whisper`.
- Timestamp-derived voiced duration, pause count, pause duration, mean and
  maximum pause, and pause ratio.
- User accounts, email verification flow, consent records, session records,
  feature-score history, and optional wellness habit check-ins.
- Trend charts after at least three sessions.
- A data-minimized optional OpenAI insight layer that receives anonymous score
  histories and a curated action library, never audio, transcripts, names, or
  email addresses.
- Docker packaging and AWS production architecture documentation.
- Automated multilingual, privacy-boundary, persistence, and pause-analysis
  tests.
- Language-matched result labels and concise explanations for English,
  Japanese, and Chinese.
- A de-identified longitudinal research CSV export and a separate restricted
  user-management export.
- AWS Secrets Manager protection for the research-admin access key.
- A data-minimized OpenAI Responses API connection using structured output,
  anonymous score histories, curated actions, and no response storage.
- Returning-user profile restoration with name-based greeting.
- Private age-range and gender fields plus aggregate profile counts for the
  research admin view.
- In-page browser microphone recording and a local four-dimension expression
  snapshot available from the first completed reflection.
- Browser-timezone daily limits with UTC timestamps, local session dates, and
  one-time correction of legacy UTC-dated sessions.
- A public `docs/` knowledge center covering product and UX learnings,
  engineering lessons, responsible wellness design, and key decisions.

### Changed

- Require one explicit daily wellness-habit selection instead of ambiguous
  independent checkboxes.
- Present the reflection flow as language, record-or-upload, and analyze steps;
  keep the eight technical features available in a compact details section.

- Replaced diagnosis-like and cognitive-age language with sample-level,
  descriptive feature language.
- Simplified the primary user journey to sign in, choose a language, select a
  recording, analyze, and review results.
- Replaced percentage-like score rendering with mobile-friendly cards showing
  explicit `0–100` sample feature scores.
- Increased the daily reflection limit from two to three so a user can unlock
  the first personal trend chart in one day.
- Separated the current implementation in `aoi_kinabot_app/` from historical
  exploratory files elsewhere in the repository.

### Privacy

- Raw audio is processed through a temporary file and deleted on success or
  failure.
- Full transcripts are not persisted.
- Feature scoring does not use OpenAI.

## [1.0.1] - 2026-01-10

- Historical repository release retained for provenance.

## [1.0.0] - 2025-12-13

- Initial historical repository release retained for provenance.
