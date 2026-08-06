# KinaBot Offline Research Mode

Offline Research Mode supports university research on one approved computer or
trusted local network. Speech processing and longitudinal records remain under
the research institution's control.

## What Offline Mode Enforces

When `KINABOT_OFFLINE_RESEARCH_MODE=true`:

- participants use a school-assigned ID such as `001`, not an email address;
- the database stores a keyed, study-specific HMAC pseudonym rather than the raw ID;
- email and SMTP verification are bypassed;
- OpenAI is disabled even if an API key is present;
- transcription requires an existing local Whisper model directory;
- KinaBot does not request a model download by name;
- feature scoring, longitudinal history, and research export stay local; and
- raw audio and full transcripts are not retained by KinaBot.

The installer generates a local participant-key secret. It must be protected
with the same care as the database and backed up securely; losing or changing it
breaks linkage to prior sessions. The institution must keep the human
participant-to-ID key separately. KinaBot's UI
may display research-source links, but it never opens them automatically.

## Prepare the Offline Package

On an approved internet-connected administration computer:

```powershell
python -m pip download -r requirements-offline.txt -d wheelhouse
```

Also obtain a compatible CTranslate2/faster-whisper model directory. Copy the
repository, wheelhouse, and model to institution-approved encrypted media.
Record file hashes, licenses, the Git commit, and the scoring version.

## Install Without Internet

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links .\wheelhouse -r requirements-offline.txt
```

Start with an explicit local model path:

```powershell
.\run-offline.ps1 -WhisperModelPath "D:\KinaBot\models\whisper-small"
```

Open `http://127.0.0.1:8501`. Participant IDs must contain 3-32 letters,
numbers, underscores, or hyphens. `001` is valid.

## Offline Acceptance Test

1. Disconnect wired, wireless, and VPN network access.
2. Start KinaBot with `run-offline.ps1`.
3. Enter test ID `001` and confirm no email or verification code is requested.
4. Analyze a non-sensitive test recording twice.
5. Confirm local results and longitudinal history.
6. Confirm no raw audio or full transcript remains after processing.
7. Export and inspect the de-identified research CSV.
8. Record the app commit, model hash/path, scoring version, OS, and result.

## Research Responsibility

Offline operation does not by itself establish GDPR compliance, clinical
validity, ethics approval, or regulatory clearance. The university remains
responsible for its lawful basis, ethics review, participant information,
retention schedule, device security, backups, deletion, incident response, and
appropriate interpretation.
