# Synthetic Test Data Guide

Use clearly fictional data in public bug reports, tests, screenshots, and
examples. This keeps contributors from accidentally publishing information
about a participant.

## Choose Synthetic Data

- **Synthetic** data is invented for the example. It does not come from a
  person, recording, transcript, account, or production system.
- **Anonymized** data started as real data and identifiers were removed. It can
  still be identifiable through voice, wording, context, or combinations of
  details.
- **Pseudonymized** data replaces a real identifier with another value. It is
  still personal data if it can be linked back to a person.
- **Redacted** data hides selected parts of real data. The remaining parts can
  still expose sensitive information.

For public contributions, use synthetic data. Removing names is not enough:
voice and language data may still identify a person.

## Make a Safe Example

Invent short, non-clinical material from scratch. Do not adapt a recording,
transcript, or production export. These examples are fictional and safe to use
as test fixtures:

- English: `The blue lantern rests beside a paper kite.`
- Japanese: `青いちょうちんが紙のたこのそばにあります。`
- Chinese: `蓝色灯笼在纸风筝旁边。`

Keep examples small. Use obvious placeholders such as `sample-user` or
`test-session-001`, not a real participant ID.

## Public Contribution Examples

### Bug report

```text
Expected the example session to render after upload.
Fixture: "The blue lantern rests beside a paper kite."
Result: the summary panel stays empty.
```

### Automated unit test

```python
sample_text = "The blue lantern rests beside a paper kite."
assert score_sample(sample_text).language == "en"
```

### Screenshot

Use a new local test session with one fictional example. Before sharing,
check that browser tabs, file paths, account names, and notification previews
do not reveal private information.

### Sanitized log

```text
INFO sample_id=test-session-001 language=en stage=render status=complete
```

Do not paste a production log. Recreate only the relevant event with invented
values.

## Never Include

Do not put any of these in a public issue, pull request, test fixture,
screenshot, log, or documentation:

- audio or recordings;
- transcripts or lightly edited participant text;
- email addresses, names, or participant IDs;
- credentials, API keys, tokens, or `.env` values;
- medical details or clinical notes;
- local databases, exports, backups, or production data.

## When Sensitive Data Is Required

Stop. Do not upload it or try to redact it for a public thread. Follow the
private reporting path in [SECURITY.md](../SECURITY.md) and share only
the minimum information needed after the maintainer confirms a safe channel.
