# Changelog

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

### Changed

- Replaced diagnosis-like and cognitive-age language with sample-level,
  descriptive feature language.
- Simplified the primary user journey to sign in, choose a language, select a
  recording, analyze, and review results.
- Replaced percentage-like score rendering with mobile-friendly cards showing
  explicit `0–100` sample feature scores.
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
