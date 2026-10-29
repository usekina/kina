# Self directed wellness actions

The UI presents descriptive speech observations separately from general daily
wellness choices. It does not interpret a feature change as a need for a food,
exercise duration or social activity. Feature-index differences remain points.
The compact care panel separately reads the raw repetition ratio, displaying
the current and previous percentages, relative percentage change and (inside
calculation details) percentage-point change. It never converts an index into
a measured percentage or interprets this as health improvement or decline.

## User journey

Results show the latest recording against its previous comparable recording
when both have measured features and matching language, scoring version and
pipeline provenance. Unknown or unavailable measurements are excluded. This
2-record descriptive comparison is not a validated personal baseline and is
separate from existing 3-record trend/reference requirements.

Results, Today and Trends expose a centered, single-column daily choice with
three main actions: social connection, comfortable physical activity and a
balanced meal. Learning and custom actions remain inside More choices. Skip
is explicit. Each primary action has one short rationale and its own collapsed
research section: I-CONECT (2024 issue), WHO (2019 edition) or MIND (2023).
The sections explain populations, scope and limitations, including MIND's null
between-group result. Sources do not validate KinaBot or establish an individual
dose or an effect on this participant's speech score. The flow is local and
self-directed, not labeled AI-personalized.

Repetition percentages use the stored `ratio` in the Repetition Pattern raw
metric: repeated token instances after first occurrence / tokenized words.
Compare the latest session with its immediately preceding compatible session,
matching language, scoring version, pipeline and session type. Unknown provenance,
unavailable scores, missing/invalid ratios and missing prior measurements show
an empty state, never fabricated data. A prior ratio of zero has a point change
but no relative percentage change. Stored ratios have three-decimal precision;
display changes inherit that precision. Topic, tokenization and recording
conditions can affect the result. No clinical interpretation is made.

Saving a choice records a plan. Completion requires a separate participant
confirmation. Replacing a plan resets it to planned; skip has no completion
button. The footer counts self-confirmed completed plans through the local day
and shows the latest completion. Replacing today's plan also replaces that
day's completion, so this is current plan status, not an immutable event log.
Historical habit check-ins remain untouched and are not reinterpreted
as plans. No streak, quota or penalty is attached to skipping.

## Data handling

An additive wellness_plans table holds one record per participant/local date:
action identifier, optional custom text (maximum 160 characters), plan status,
and update time. Account export includes these records; account deletion removes
them. No change is made to existing research-record export. The UI discourages
identifying or medical content in optional text. Apply the existing protected
account-storage, backup, retention and deletion procedures to these records.

No OpenAI request is made to select wellness actions. The old insight helper is
retained as a local compatibility function but is no longer used by the app.
It returns a neutral menu example independent of scores. This change introduces
no paid tier, new vendor, prompt, clinical recommendation or medical claim.
Future preference-based AI assistance needs separate scope and review.

## Evidence and validation

Content reference:
https://www.nia.nih.gov/health/cognitive-health-and-older-adults

Primary action references (reviewed 2026-10-29):
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10943511/
- https://www.who.int/publications/i/item/9789241550543 (2019 edition)
- https://pubmed.ncbi.nlm.nih.gov/37466280/

Tests cover action invariance when scores change, no API calls even when a key
is set, comparable-session selection, exclusion of unavailable measurements,
plan versus completion semantics, custom length validation, account/date
isolation, export/deletion, and the EN/JA/ZH form flows with Streamlit AppTest.
Preview uses synthetic scores only. Production authentication and deployed-device
checks remain required. No release or cloud deployment is performed by this PR.

Before rollout, back up the participant database and verify account export and
deletion on staging. Older application code ignores the additional table;
rollback does not require dropping it or restoring a stale database.
