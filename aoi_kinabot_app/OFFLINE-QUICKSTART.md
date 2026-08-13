# KinaBot Offline Research — Quick Start

> The public GitHub repository does not include `wheelhouse/` or
> `models/whisper-small/`. These large, versioned third-party artifacts belong
> to a separately built and checksum-verified institutional delivery bundle.
> If you received only a GitHub link, request the complete offline bundle from
> the KinaBot maintainer before starting installation.

## For the Study Administrator

1. Copy the complete approved KinaBot offline bundle to the encrypted research
   computer.
2. Confirm these folders exist:
   - `wheelhouse/`
   - `models/whisper-small/`
3. Open PowerShell in this folder.
4. Run `Set-ExecutionPolicy -Scope Process Bypass` if institution policy allows.
5. Run `.\install-offline.ps1` once.
6. Disconnect the computer from the network if required by the protocol.
7. Run `.\run-offline.ps1`.
8. Open `http://127.0.0.1:8501`.

If approved university software will call KinaBot directly, run
`.\offline_api\run-offline-api.ps1` instead and open
`http://127.0.0.1:8787/docs`. See the
[Offline/Private Research API guide](offline_api/README.md). The local API uses
the same analysis engine and database; it does not call OpenAI.

## For a Participant

1. Enter the ID assigned by the university, for example `001`.
2. Select English, Japanese, or Chinese.
3. Read and accept the study/product consent shown by the approved protocol.
4. Select the approved audio recording.
5. Review the descriptive speech features.

Do not enter a name or email. The research team must keep the separate ID key
outside KinaBot. KinaBot does not diagnose, screen for disease, or provide
medical advice.

## Stop the App

Return to the PowerShell window and press `Ctrl+C`.

For installation validation, data governance, and troubleshooting, see
`docs/OFFLINE-RESEARCH-GUIDE.md` and
`docs/UK-EU-DATA-PROTECTION-READINESS.md`.
