# KinaBot Offline/Private Research API

This package lets approved university software call KinaBot on the same Windows
computer without internet, email, cloud transcription, or OpenAI. It is an
integration surface for the existing versioned KinaBot feature engine—not a
second scoring implementation and not a diagnostic API.

## Trust Boundary

- Binds to `127.0.0.1` by default; it is not exposed to the campus network.
- Requires a locally generated `X-KinaBot-Token` for every data operation.
- Converts a university-assigned ID into a study-secret HMAC pseudonym. The raw
  ID is not stored or returned.
- Uses the preinstalled local `faster-whisper` model and KinaBot's multilingual
  Python/NLP scoring.
- Deletes temporary audio on success and failure; does not persist full
  transcripts.
- Removes `OPENAI_API_KEY` at launch. Calls create no OpenAI usage or AImoji API
  bill.
- Returns versions, retention statements, self-comparison scope, and an
  explicit non-diagnostic boundary with every analysis.

These controls support but do not replace university ethics, information
governance, legal, accessibility, security, and study-protocol approval.

## Administrator Quick Start

Use the complete approved offline bundle. From `aoi_kinabot_app`:

```powershell
.\install-offline.ps1
.\offline_api\run-offline-api.ps1
```

Open `http://127.0.0.1:8787/docs` for the generated OpenAPI interface and
`http://127.0.0.1:8787/health` for readiness. The calling application reads the
local token from `.offline-api-token`; keep that file on the encrypted research
computer and never email it, commit it, or include it in screenshots.

The API and Streamlit research UI use the same database by default. Run only
the surfaces approved by the study protocol.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Local readiness and version metadata; no participant data |
| `POST` | `/v1/reflections` | Transcribe and calculate one descriptive feature set |
| `GET` | `/v1/participants/{id}/history` | Return that participant's derived longitudinal observations |
| `DELETE` | `/v1/participants/{id}` | Delete that participant's local derived record |

Example PowerShell request:

```powershell
$token = (Get-Content .\.offline-api-token -Raw).Trim()
$headers = @{ "X-KinaBot-Token" = $token }
$form = @{
    participant_id = "001"
    language = "English"
    consent_version = "approved-study-consent-v1"
    session_type = "weekly-reflection"
    audio = Get-Item "C:\approved-study-input\sample.wav"
}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8787/v1/reflections" `
    -Headers $headers -Form $form
```

Supported language values are `English`, `日本語`, and `中文`. Supported local
transcription formats are MP3, MP4, MPEG, MPGA, M4A, WAV, and WEBM.

## Response Contract

An analysis contains feature evidence plus:

```json
{
  "provenance": {
    "app_version": "v1.2-offline-research",
    "scoring_version": "score-v2-multilingual",
    "transcription": "local-faster-whisper"
  },
  "interpretation": {
    "comparison_scope": "self_only",
    "non_diagnostic": true
  },
  "retention": {
    "audio": "ephemeral",
    "full_transcript": "not_stored"
  }
}
```

Clients must preserve these limitations when presenting results. They must not
rename a feature as a diagnosis, disease screen, clinical measurement, or risk.

## Expense and Commercial Boundary

Offline use does not consume an OpenAI API key and has no per-request OpenAI
charge. The institution supplies its computer, storage, electricity, IT
support, and approved local model bundle. Open-source availability does not
promise free deployment, customization, validation, training, maintenance, or
future institutional service. Those require a separate written agreement where
applicable. Repository licensing and commercial terms should be reviewed before
any production or redistribution commitment.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
    test_multilingual.py test_offline_mode.py offline_api\tests
```

The tests cover authorization, pseudonymization boundaries, non-retention in
responses, version provenance, deletion, invalid IDs, unsupported files, and
the existing multilingual/offline behavior. They establish engineering
behavior—not clinical validity, legal compliance, institutional approval, or
real-world effectiveness.

## Pilot Evidence to Preserve

With participant consent and university approval, preserve non-sensitive proof
of institutional evaluation separately from participant data:

- dated installation and acceptance record;
- approved study protocol/version and institutional contact;
- software commit and bundle checksum used;
- number of completed sessions in aggregate;
- usability and integration feedback;
- defects, resolutions, and validation results; and
- publication, citation, adoption, or continuation decisions when they occur.

Never place participant audio, transcripts, raw IDs, secrets, private exports,
or unapproved university correspondence in GitHub.
