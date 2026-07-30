# Product and UX Learnings

## Product Goal

KinaBot should let an ordinary person complete a short voice reflection with
minimal instruction, receive an understandable result immediately, and return
over time without feeling tested or judged.

## 1. Record in the Page

**Observed problem:** Asking mobile users to record in another app, locate the
file, and upload it creates avoidable drop-off. Some iPhone users do not know
where a recording was saved.

**Design response:** Make browser recording the primary path:

1. choose the spoken language;
2. choose **Record now** or **Upload a recording**; and
3. select **Analyze my reflection**.

Upload remains available for accessibility, testing, and previously recorded
audio. Browser recording requires HTTPS and a clear microphone-permission
prompt.

## 2. Give Value on the First Session

**Observed problem:** Eight technical feature scores and a disclaimer can feel
like work without a useful outcome.

**Design response:** Present a friendly first-session expression snapshot:

- Expression;
- Flow;
- Clarity; and
- Vocal energy.

These four dimensions are transparent presentations of existing local features,
not new medical or personality measurements. The detailed eight-feature view
remains available in an expandable section.

Every completed reflection should provide:

- one core takeaway;
- one small action to try tomorrow; and
- a clear explanation that trends unlock after repeated sessions.

## 3. Avoid Entertainment Metrics That Pretend to Be Science

“Luck,” “fortune,” intelligence, cognitive age, and disease risk may attract
attention but cannot be responsibly inferred from a short recording.

Friendly design is encouraged. Unsupported interpretation is not. Emotional
wording may describe language used in a sample, but it must not claim to know a
person's internal emotional state.

## 4. Remember Returning Users

**Observed problem:** Re-entering name and profile information makes the app
feel forgetful and increases burden.

**Design response:** Use the verified email as the account key. Collect the
required pilot profile once, store it privately, restore it on later visits, and
greet the returning user by name. Keep account settings available but collapsed.

## 5. Use the User's Local Calendar Day

**Observed problem:** A server running in UTC can count an evening U.S. session
as the next day.

**Design response:** Keep the precise event timestamp in UTC, but calculate the
daily usage date from the browser's IANA timezone. Reset the three-session
allowance at the user's local midnight.

This is easier to understand than a rolling 24-hour window and preserves an
auditable explanation of how each session date was assigned.

## 6. Make Habit Check-ins Unambiguous

Independent checkboxes made it unclear whether zero, one, or several habits
were expected.

The pilot now asks the user to select one self-reported wellness habit for the
day. Habit records stay separate from speech scores, and KinaBot does not claim
that a habit caused a score change.

## 7. Match the User's Language

Interface labels, explanations, takeaways, and actions should match the language
selected for the recording. Multilingual support is not word-for-word
translation: segmentation, pace units, grammar cues, examples, and validation
must be language appropriate.

## 8. Reduce Words, Not Transparency

The main path should be short and calm. Privacy, scoring details, and boundaries
should remain available through concise cards and expandable detail sections.
Critical consent must never be hidden.

## Pilot Questions to Measure

- Can a new mobile user finish without help?
- How long does the first reflection take?
- Does the first result feel useful?
- Does the user understand that scores describe one sample?
- Does the user return within seven days?
- Where do users abandon the flow?
- Do English, Japanese, and Chinese users interpret the labels similarly?

Record measured answers in `IMPACT.md`; do not replace evidence with promotional
claims.
