# V1.1.0 Release Checklist

## Engineering

- [x] English, Japanese, and Chinese NLP paths
- [x] Private/local transcription path
- [x] Timestamp-based pause metrics
- [x] Versioned score persistence
- [x] Trends after three sessions
- [x] Automated tests and GitHub Actions
- [ ] Real English recording test
- [ ] Real Japanese recording test
- [ ] Real Chinese recording test
- [ ] Production database migration
- [ ] Account and history deletion

## Privacy and Safety

- [x] No persistent raw audio
- [x] No persistent full transcript
- [x] OpenAI excluded from transcription and scoring
- [x] Anonymous score-only insight contract
- [ ] Final public privacy notice reviewed
- [ ] Production retention and deletion procedure tested
- [ ] Research/pilot consent reviewed separately

## Operations

- [x] Dockerfile
- [x] Health-check endpoint configuration
- [x] AWS ECS/ALB/EFS staging deployment
- [ ] HTTPS and stable production URL
- [ ] Backup and restore test
- [ ] Logging, monitoring, and alerting
- [ ] Limited 5–10 user pilot before broader access

## Evidence

- [x] Maintainer and historical-work boundary
- [x] Versioned methodology
- [x] Public changelog
- [x] Verifiable impact template
- [ ] Store release test evidence in the private archive
- [ ] Create GitHub release after this checklist's required V1.1 items pass
