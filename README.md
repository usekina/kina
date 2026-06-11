<img width="200" height="200" alt="Kina logo" src="https://github.com/user-attachments/assets/6101cefa-ff62-4511-a290-0b3dba87160d" />

# KinaBot

**A dignity-first speech reflection tool for older adults, families, and caregivers.**

KinaBot helps users and families notice possible changes in speech and communication patterns through natural conversation analysis. It is designed to support awareness, reflection, and better conversations with care professionals.

KinaBot is **not a medical device**, **not a diagnostic tool**, and **not a replacement for licensed healthcare professionals**.

<img width="462" height="235" alt="image backed by-1229" src="https://github.com/user-attachments/assets/ea03453b-df66-4561-b5bd-719c2f63e27c" />

<img width="720" height="278" alt="1229 team background" src="https://github.com/user-attachments/assets/514f3348-2d1f-4aa4-bea5-7dbd46fbddd3" />
<img width="780" height="329" alt="team 4 people" src="https://github.com/user-attachments/assets/47648fa8-7225-4ae1-af25-467ede08b849" />

<a href="https://www.producthunt.com/products/kinabot?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-kizuna" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=960544&theme=light&t=1764382304451" alt="Kina - From words to wellbeing" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Mission

KinaBot is built around a simple belief: technology for aging should protect human dignity, strengthen family connection, and help people seek support earlier without fear or stigma.

The project focuses on:

- Consent-based speech reflection
- Communication pattern awareness
- Family and caregiver support
- Privacy-aware design
- Clear boundaries between wellness insight and medical diagnosis

## What KinaBot Does

The current prototype can:

- Record short speech samples
- Convert speech to text
- Analyze basic language patterns such as word variety, sentence structure, speaking pace, and emotional tone
- Generate a simple report for personal reflection or family discussion
- Support English and Japanese speech recognition in the current interface

## What KinaBot Does Not Do

KinaBot does not:

- Diagnose dementia, cognitive impairment, or any medical condition
- Replace doctors, clinicians, therapists, or professional cognitive assessment
- Provide emergency support
- Guarantee that a speech pattern means a health condition exists
- Determine medical risk without clinical review

If you or a family member has health concerns, please consult a qualified healthcare professional.

## Data And Privacy

KinaBot works with sensitive voice and language data. Users should only record or upload speech when they have proper consent from every person whose voice may be included.

Current prototype behavior:

- Audio may be processed for speech-to-text transcription.
- The current Python prototype uses the `SpeechRecognition` library and may use Google speech recognition services unless replaced by a local or private transcription backend.
- Recordings may be saved locally by the prototype during analysis.
- Generated reports may include transcripts and speech pattern summaries.

Before using KinaBot with real users, review [Privacy.md](Privacy.md), [Medical-Disclaimers.md](Medical-Disclaimers.md), and [LEGAL-DISCLAIMER.md](LEGAL-DISCLAIMER.md).

## Responsible Use Principles

KinaBot should be used with:

- Clear user consent
- Respect for older adults' autonomy
- Minimal data collection
- Transparent explanation of what is analyzed
- Human review before any care decision
- Professional medical consultation when concerns arise

## Project Status

KinaBot is an early-stage source-available prototype for research, education, and non-commercial exploration. The project is not clinically validated and should not be used as a standalone medical screening system.

## License

This repository uses a non-commercial license. Personal, academic, and non-profit research use is allowed under the license terms. Commercial use requires prior written permission.

See [LICENSE.md](LICENSE.md) for details.
