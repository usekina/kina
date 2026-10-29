# Self directed wellness actions

The UI presents descriptive speech observations separately from general daily
wellness choices. It does not interpret a feature change as a need for a food,
exercise duration or social activity. Differences are feature-index points,
never a percentage or evidence of health improvement or decline.

## User journey

Results show the latest recording against its previous comparable recording
when both have measured features and matching language, scoring version and
pipeline provenance. Unknown or unavailable measurements are excluded. This
2-record descriptive comparison is not a validated personal baseline and is
separate from existing 3-record trend/reference requirements.

Results, Today and Trends expose the same daily choice: social connection,
comfortable physical activity, a balanced meal, an interest/learning activity,
a custom action, or skip. The evidence expander links to the NIA public guide
and explains its general scope. The source does not validate KinaBot or establish
an individual dose or an effect on this participant's speech score.

Saving a choice records a plan. Completion requires a separate participant
confirmation. Replacing a plan resets it to planned; skip has no completion
button. Historical habit check-ins remain untouched and are not reinterpreted
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

Tests cover action invariance when scores change, no API calls even when a key
is set, comparable-session selection, exclusion of unavailable measurements,
plan versus completion semantics, custom length validation, account/date
isolation, export/deletion, and the EN/JA/ZH form flows with Streamlit AppTest.
Preview uses synthetic scores only. Production authentication and deployed-device
checks remain required. No release or cloud deployment is performed by this PR.

Before rollout, back up the participant database and verify account export and
deletion on staging. Older application code ignores the additional table;
rollback does not require dropping it or restoring a stale database.
