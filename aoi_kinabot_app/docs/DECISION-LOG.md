# KinaBot Decision Log

This log records important product and engineering decisions. Dates describe
when the decision was documented, not necessarily the first time an idea was
considered.

| Date | Decision | Reason |
|---|---|---|
| 2026-07-28 | Keep feature scoring in local Python/NLP | Reproducibility, privacy, and clear ownership of the core method |
| 2026-07-28 | Support English, Japanese, and Chinese with language-specific adapters | A global tool cannot treat English rules as universal |
| 2026-07-28 | Do not retain raw audio or full transcripts | Reduce sensitivity and align retention with the product's actual need |
| 2026-07-28 | Limit the LLM to anonymous score trends and curated actions | Prevent the LLM from becoming the scoring or diagnostic authority |
| 2026-07-29 | Allow three reflections per local day | Let users see an early trend while limiting repetitive testing and cost |
| 2026-07-29 | Restore returning-user profiles by verified email | Remove repeated onboarding burden |
| 2026-07-29 | Require one daily habit selection | Make self-reported habit data easier to understand |
| 2026-07-29 | Add direct browser recording with upload fallback | Reduce mobile friction without removing accessibility options |
| 2026-07-29 | Add a four-dimension first-session expression snapshot | Give an understandable result before longitudinal trends exist |
| 2026-07-29 | Use CloudFront HTTPS | Enable secure mobile microphone access with a stable AWS endpoint |
| 2026-07-29 | Store UTC timestamps plus browser-local dates and IANA timezone | Reset limits at the user's expected midnight and preserve auditability |
| 2026-07-30 | Establish a public knowledge center | Share verified product and engineering lessons while protecting users and invention-sensitive details |
| 2026-08-02 | Make the latest eight features and Trends reachable from mobile navigation | Reduce portrait scrolling and let returning users revisit results without repeating an analysis |
| 2026-08-02 | Publish anonymized feedback and scoring explanations | Make design reasoning educational and auditable without exposing participant identity or overstating evidence |
| 2026-08-03 | Introduce a low-pressure 30-day pattern experience with one recommended daily reflection | Make the value understandable before asking for repeated use; keep extra check-ins optional and avoid streak penalties |
| 2026-08-05 | Establish validation gates, an active AI risk register, and evidence-defined impact metrics | Make future releases, pilots, public claims, and independent evaluation follow one auditable human-centered framework |
| 2026-08-05 | Create an offline university research mode with school IDs, study-secret HMAC pseudonyms, and mandatory local transcription | Support privacy-sensitive UK and university research without email, cloud scoring, or silent model downloads |
| 2026-08-06 | Publish meaningful changes as dated, evidence-linked learning cases | Preserve first-hand failures, tradeoffs, verification, and social context as reusable education rather than retrospective promotion |

## Template for Future Decisions

Add a row above and, when more detail is useful, include:

- problem observed;
- alternatives considered;
- decision;
- evidence;
- risks;
- validation needed; and
- version or pull request that implemented the change.
