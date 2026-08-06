# Offline Research Bundle Manifest

Every university delivery must identify:

- KinaBot Git commit and release;
- application and scoring-model versions;
- supported operating system and Python version;
- exact wheelhouse contents and licenses;
- faster-whisper/CTranslate2 model name, source, version, license, and SHA-256;
- bundle SHA-256 manifest;
- build date and builder;
- institution and approved research purpose;
- offline acceptance-test result;
- known limitations; and
- contact for security or data-protection issues.

`build-offline-bundle.ps1` creates a commit-specific ZIP and file manifest from
an approved wheelhouse and model directory. `verify-offline-bundle.ps1` checks
that required components exist and that every manifested file is unchanged.

The generated bundle, model, wheelhouse, participant-key secret, databases, and
research exports must not be committed to the public repository.
