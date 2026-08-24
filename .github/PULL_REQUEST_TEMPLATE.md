## Summary

<!-- What changed, why, and for whom? Keep this outcome-focused. -->

## Related issue

<!-- Use "Fixes #123" only when merging this PR should close the entire issue. -->

## Scope

<!-- State what is included and intentionally out of scope. -->

<!-- Changes under aoi_kinabot_app/ must be authored and committed by Aoi using an approved identity. External contributors should use an issue or patch proposal for that area. -->

## Verification

<!-- List exact automated commands and manual user journeys with results. -->

- [ ] `python -m pytest`
- [ ] `python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
- [ ] Focused regression or acceptance tests added/updated
- [ ] Relevant manual flow verified, or not applicable

## Evidence

<!-- Add before/after screenshots for UI work and sanitized logs where useful. Never include participant or production data. -->

## Impact review

For each item, describe the impact below or write **Not applicable**.

- **Scoring definition/version and historical trends:**
- **Privacy, consent, retention, deletion, or temporary files:**
- **Security and abuse resistance:**
- **Accessibility and mobile behavior:**
- **Multilingual behavior:**
- **API/data migration and backward compatibility:**
- **AWS deployment, operations, observability, and cost:**
- **New or changed dependencies:**

## Documentation and release notes

- [ ] User, scoring, API, operations, or deployment documentation updated
- [ ] Changelog/release note added for a notable user-visible or operational change
- [ ] No documentation change is needed (explain why)

## Responsible-use checklist

- [ ] No diagnosis, disease prediction, cognitive-age, treatment, or unsupported clinical claims are introduced.
- [ ] No secrets, personal data, participant audio, transcripts, or confidential material are committed.
- [ ] Commit author and committer metadata use a personal or explicitly authorized project email, not an unrelated employer domain.
- [ ] Raw audio and temporary sensitive files are not stored by default and cleanup still occurs on failure.
- [ ] New data collection has an explicit purpose, consent path, retention rule, and deletion path.
- [ ] Synthetic data is used in tests, examples, screenshots, and logs.
- [ ] Limitations and uncertainty are visible where users could otherwise overinterpret results.

## Breaking change or rollout plan

<!-- Describe migrations, compatibility, rollback, deployment sequence, and post-deployment checks. Write "Not applicable" when appropriate. -->

## Contributor statement

- [ ] I have the right to submit this work and it is compatible with the repository license.
- [ ] I have read and agree to the prospective contribution terms in `CLA.md` for this contribution.
- [ ] I have read `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- [ ] I understand that merge does not by itself mean AWS deployment or real-user validation.
