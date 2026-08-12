# KinaBot System Architecture

**Scope:** Aoi-maintained `aoi_kinabot_app/`
**Purpose:** Public technical reference for builders, reviewers, and students

KinaBot is a dignity-first, multilingual, privacy-aware system for longitudinal
speech reflection. It describes observable speech and language features; it is
not a diagnostic system.

## Processing and interpretation architecture

```mermaid
flowchart LR
    subgraph U["User-controlled input"]
        U1["Voice reflection<br/>phone or computer"]
        U2["English, Japanese, or Chinese"]
        U3["Consent and context"]
    end
    subgraph P["Privacy-aware processing"]
        P1["Temporary audio copy"]
        P2["Private or local Whisper<br/>transcription and timestamps"]
        P3["Delete temporary audio"]
        P4["Discard full transcript"]
    end
    subgraph E["Versioned KinaBot engine"]
        E1["Language-specific adapter"]
        E2["Observable feature extraction"]
        E3["Deterministic scoring"]
        E4["Version provenance"]
    end
    subgraph D["Data-minimized record"]
        D1["Pseudonymous participant or account"]
        D2["Session metadata"]
        D3["Derived metrics and scores"]
    end
    subgraph X["Human-centered experience"]
        X1["Single-session description"]
        X2["Self-only longitudinal trend"]
        X3["Review and deletion"]
    end
    subgraph B["Evidence boundary"]
        B1["Recorded observation"]
        B2["Descriptive pattern"]
        B3["Validated interpretation"]
        B4["No diagnosis or cognitive-decline claim"]
    end
    U1 --> P1 --> P2 --> E1 --> E2 --> E3 --> E4
    U2 --> E1
    U3 --> D1
    P1 --> P3
    P2 --> P4
    E4 --> D2
    E4 --> D3
    D2 --> X1
    D3 --> X1
    D3 --> X2
    X1 --> X3
    X2 --> X3
    X1 --> B1
    X2 --> B2
    B1 --> B2
    B2 -. "independent evidence required" .-> B3
    B3 --> B4
```

The source recording remains under the user's control. KinaBot retains the
minimum derived record required for the selected deployment and exposes the
method and version provenance needed to interpret it responsibly.

## Shared core, different trust boundaries

```mermaid
flowchart TB
    K["Shared KinaBot core<br/>multilingual feature engine<br/>versioned scoring<br/>claims boundaries"]
    subgraph O["Online public service"]
        O1["Verified user access"] --> O2["HTTPS application"]
        O2 --> O3["Protected persistent storage"]
        O2 --> O4["Public usage policy"]
        O3 --> O5["Optional data-minimized insight"]
    end
    subgraph R["Offline / Private Research API"]
        R1["University research ID"] --> R2["Study-secret HMAC pseudonym"]
        R2 --> R3["Local Windows computer"]
        R3 --> R4["localhost API"]
        R4 --> R5["Local Whisper"]
        R4 --> R6["Protocol-configurable policy"]
        R5 --> R7["No email, internet, or OpenAI required"]
    end
    K --> O
    K --> R
```

The offline API is an integration surface for the same versioned engine, not a
second scoring implementation. Deployment changes identity, network, storage,
and governance requirements; it does not change the evidence boundary.

## Responsibility boundary

| Responsibility | System owner |
|---|---|
| Speech-to-text | Approved private/local transcription engine |
| Language processing and feature extraction | KinaBot language adapters |
| Feature-score calculation | Versioned KinaBot scoring engine |
| Personal trend calculation | KinaBot application |
| Optional plain-language action | Data-minimized insight layer or curated fallback |
| Clinical meaning | Not established by the current system |
| Study interpretation | Approved protocol and qualified independent review |

Architecture diagrams document intended boundaries; tests and deployment
records must verify actual behavior. Use the [Benchmark and Evaluation
Framework](../evaluation/BENCHMARK-FRAMEWORK.md) for structured evaluation.
