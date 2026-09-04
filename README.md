<img width="200" height="200" alt="Kina logo" src="https://github.com/user-attachments/assets/6101cefa-ff62-4511-a290-0b3dba87160d" />

# KinaBot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22305527.svg)](https://doi.org/10.5281/zenodo.22305527)

> [!IMPORTANT]
> **This is the official, actively maintained KinaBot repository.** KinaBot is
> developed and maintained by Aoi Minamoto through AImoji LLC. The earlier
> [`aoiminamoto/kina`](https://github.com/aoiminamoto/kina) repository is
> retained only as a historical record. Active releases, issues, security
> reports, and contributions belong here.

> **KinaBot is a dignity-first, privacy-aware AI system for longitudinal speech
> reflection, healthy aging, and family-centered care.**

KinaBot helps people reflect on observable speech and communication patterns
across repeated, natural-language samples. It is designed to support personal
awareness, family connection, and more informed conversations with care
professionals while preserving the older adult's autonomy.

KinaBot is **not a medical device**, **not a diagnostic tool**, and **not a replacement for licensed healthcare professionals**.

## Current Ownership And Maintainership

KinaBot is currently maintained by Aoi Minamoto through AImoji LLC.

Earlier exploratory materials may remain in this repository for transparency and project history. They do not indicate current ownership, employment, maintainership, or product rights.

Current Aoi-maintained development is located in `aoi_kinabot_app/`.

## Current Development Status

This repository preserves earlier exploratory work and product-direction discussions for transparency. Current development focuses on a new KinaBot implementation maintained by Aoi Minamoto through AImoji LLC. Future public-facing releases will be developed separately from earlier exploratory materials, with clearer safety, privacy, and responsible-use boundaries.

Current Aoi-maintained development is in `aoi_kinabot_app/`.

## Participate

KinaBot welcomes code, tests, documentation, translation, accessibility review,
privacy review, reproducible research, issue triage, and product feedback.

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security and private vulnerability reporting](SECURITY.md)
- [Product and engineering roadmap](ROADMAP.md)
- [Release process](RELEASING.md)
- [Current application changelog](aoi_kinabot_app/CHANGELOG.md)
- [Scientific evidence and claims boundaries](aoi_kinabot_app/docs/SCIENTIFIC-EVIDENCE-AND-CLAIMS.md)
- [Institutional pilot evidence framework](aoi_kinabot_app/docs/institutional-pilots/README.md)
- [How to cite KinaBot](CITATION.cff)
- [Licensing guide](LICENSING.md) and [commercial licensing](COMMERCIAL-LICENSING.md)
- [Community/commercial boundary](COMMUNITY-AND-COMMERCIAL.md)
- [Contributor agreement](CLA.md), [trademark policy](TRADEMARKS.md), and
  [authorship provenance](PROVENANCE.md)

Start with a scoped [issue](https://github.com/usekina/kina/issues/new/choose)
or review tasks labeled [`good first issue`](https://github.com/usekina/kina/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

KinaBot V1 does not use cognitive age, biological age, dementia risk, medical risk labels, or diagnosis-like composite scores. V1 focuses on separate speech and language feature scores, trend reflection, consent-first access, and privacy-aware data handling.

Please treat files outside `aoi_kinabot_app/` as historical exploratory materials unless otherwise stated.

The historical materials remain public to preserve provenance and show the
project's continuing technical and responsible-design evolution. The current
[changelog](aoi_kinabot_app/CHANGELOG.md),
[decision log](aoi_kinabot_app/docs/DECISION-LOG.md), and Git history document
when major design boundaries changed and why.

<img width="462" height="235" alt="image backed by-1229" src="https://github.com/user-attachments/assets/ea03453b-df66-4561-b5bd-719c2f63e27c" />

<a href="https://www.producthunt.com/products/kinabot?embed=true&utm_source=badge-featured&utm_medium=badge&utm_source=badge-kizuna" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=960544&theme=light&t=1764382304451" alt="Kina - From words to wellbeing" style="width: 250px; height: 54px;" width="250" height="54" /></a>

## Mission

KinaBot is built around a simple belief: technology for aging should protect human dignity, strengthen family connection, and help people seek support earlier without fear or stigma.

The project focuses on:

- **Dignity first:** preserve each person's agency and avoid reducing anyone to
  a score or label.
- **Privacy aware:** minimize sensitive voice and identity data throughout the
  product lifecycle.
- **Longitudinal reflection:** describe patterns across time instead of drawing
  conclusions from a single sample.
- **Healthy aging:** support everyday reflection and constructive conversations,
  without diagnosis or disease-risk claims.
- **Family-centered care:** help people share understandable observations with
  trusted family members, caregivers, and professionals on their own terms.
  
## Naming

The official product name is **KinaBot**. **Kina** is the short name used throughout the project. The project was originally launched under the name **Kizuna** in May 2025, and some early public references may still use that name.

## What KinaBot V1 Focuses On

The current Aoi-maintained V1 direction focuses on:

- Record short speech samples
- Convert speech to text
- Analyze basic language patterns such as word variety, sentence structure, speaking pace, and emotional tone
- Generate a simple report for personal reflection or family discussion
- Support English and Japanese speech recognition in the current interface

## Product Boundaries

KinaBot is focused on healthy aging and family-centered care. Manufacturing,
workplace monitoring, employee assessment, and employment decision-making are
outside the product's scope.

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

Current V1 design direction:

- Audio may be processed for speech-to-text transcription.
- The current local Python implementation uses the `SpeechRecognition` library and may use Google speech recognition services unless replaced by a local or private transcription backend.
- In local development, recordings may be temporarily saved during analysis.
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

KinaBot is an early-stage source-available public-benefit project for research, education, and non-commercial exploration. The project is not clinically validated and should not be used as a standalone medical screening system.

## Citation

For reproducible work, cite the exact release used. The current archived release is:

> Minamoto, A. (2026). *KinaBot* (Version v1.1.0) [Computer software].
> Zenodo. https://doi.org/10.5281/zenodo.22305528

Use the [KinaBot concept DOI](https://doi.org/10.5281/zenodo.22305527) when
referring to the project across all versions. Citation metadata is also available
in [CITATION.cff](CITATION.cff).

## License

This repository currently uses a custom non-commercial source-available
license. Personal, academic, and non-profit research use is allowed under its
terms. Commercial use requires prior written permission. Because the license
restricts fields of use, it is not an OSI-approved open-source license; please
describe the current project as **source-available**, not OSI open source.

See the controlling [LICENSE.md](LICENSE.md), the plain-language
[licensing guide](LICENSING.md), and the
[commercial licensing process](COMMERCIAL-LICENSING.md) for details. The same
license is recorded under Rights on the
[Zenodo v1.1.0 record](https://zenodo.org/records/22305528).
