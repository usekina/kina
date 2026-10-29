# Display and speaking languages

The opening page has two independent choices: display language and speaking
language. A short localized example explains mixed use, such as reading the
Chinese interface and speaking Japanese. Currently supported languages remain
English, Japanese and Chinese; no automatic language detection is introduced.

The display choice controls navigation, instructions and results presentation.
The speaking choice controls transcription language, language-aware analysis
and the language stored with the recording. Uploaded audio follows the same
explicit speaking choice. Users should select the language actually spoken.
The analysis area confirms both current selections before submission.

`recording_language` is durable session state rather than a widget-owned key.
Landing and recording selectors copy changes into it. Changing the interface
or navigating to Results/History does not silently reset this preference.
Preferences are not newly persisted to the account or browser across sessions.
History comparability remains grouped by speech language and analysis method.

Tests cover all nine display/speaking combinations, display changes, landing
to recording transitions, widget cleanup while navigating away, returning,
speaking changes, transcription-code selection and unsupported state defaults.
No authentication, consent, audio retention or scoring behavior is changed.
