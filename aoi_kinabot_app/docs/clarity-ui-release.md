# Clarity UI and owner administration

The UI now separates recording, saved results and history. Recording is the
default input method. Saved results show eight descriptive feature indexes on
a responsive radar, with an optional recent reference from the three preceding
compatible sessions. The current session is excluded. Missing values, emotion
fallbacks and invalid values are not plotted as zero. Reference values require
three valid measurements per feature. A recent mean is not a validated personal
baseline, and shape area is not a health score or evidence of a change point.

## Administrator access

Set `KINABOT_ADMIN_EMAIL` to the owner's exact email in the task environment.
Keep the existing `KINABOT_ADMIN_KEY` secret. The owner must supply this key during
email/code login. Only that authenticated session gets an Admin navigation item.
Knowing the email or a staging code alone is insufficient. An empty owner email
disables administrator access. Offline participant login cannot grant access.
Log out and back in after deployment to authenticate as the owner.

The existing staging login behavior for ordinary users is unchanged. It is not
proof of email ownership. Formal email delivery should be configured separately
before relying on email verification as a security boundary.

## Storage and presentation

`users.id -> test_sessions.user_id -> feature_scores.test_session_id`.
Each completed session has eight feature rows, uniquely keyed by session and
feature. No database schema migration or score recalculation is part of this UI
release. Existing records are retained. Administrator tables group on session ID,
not date/time: one row per recording, eight feature columns. Raw metrics,
availability and failure reasons remain visible in the recording detail table.
The all-account exports retain both session-level and feature-level formats.

## Validation

Run `python -m pytest aoi_kinabot_app` from an environment with the app and test
dependencies installed. Tests cover current-session exclusion, language/version/
pipeline matching, missing features, default emotion values, owner authorization,
and distinct sessions sharing the same timestamp. UI smoke checks additionally
exercise the three views in English, Japanese and Chinese, and owner login with
correct and incorrect test keys against a separate database.
