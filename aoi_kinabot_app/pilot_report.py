"""Generate a small, privacy-conscious PDF for a participant's own results."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Iterable, Mapping


def build_personal_pdf_report(
    profile: Mapping[str, object] | None,
    score_rows: Iterable[Mapping[str, object]],
) -> bytes:
    """Return a basic PDF containing only the participant's own summary.

    The report intentionally excludes raw audio, prompts, model internals,
    administrator fields, and other participants' data.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:  # pragma: no cover - exercised in deployment checks
        raise RuntimeError("PDF reports require the 'reportlab' package") from exc

    rows = [dict(row) for row in score_rows]
    latest_session = rows[-1].get("session_id") if rows else None
    latest = [row for row in rows if row.get("session_id") == latest_session]
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="KinaBot Research Pilot Report",
        author="AImoji LLC",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("KinaBot Research Pilot Report", styles["Title"]),
        Paragraph(
            "Personal informational report · Not a medical or diagnostic report",
            styles["Normal"],
        ),
        Spacer(1, 10),
    ]
    display_name = str((profile or {}).get("display_name") or "Participant")
    story.append(Paragraph(f"Participant: {display_name}", styles["Normal"]))
    story.append(
        Paragraph(
            "Generated: "
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            styles["Normal"],
        )
    )
    if not latest:
        story.append(Spacer(1, 12))
        story.append(Paragraph("No completed score session is available yet.", styles["Normal"]))
    else:
        session = latest[0]
        story.extend(
            [
                Spacer(1, 10),
                Paragraph(
                    "Latest session: "
                    + str(session.get("session_date") or session.get("created_at") or "—"),
                    styles["Normal"],
                ),
                Paragraph("Language: " + str(session.get("language") or "—"), styles["Normal"]),
                Paragraph(
                    "Scoring model: " + str(session.get("scoring_model_version") or "—"),
                    styles["Normal"],
                ),
                Spacer(1, 8),
            ]
        )
        table_data = [["Feature", "Score"]]
        for row in latest:
            table_data.append([str(row.get("feature_name") or "—"), str(row.get("score") or "—")])
        table = Table(table_data, colWidths=[4.9 * inch, 1.0 * inch], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e85d2a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9dee8")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Completed sessions in this account: {len({r.get('session_id') for r in rows})}", styles["Normal"]))
    story.extend(
        [
            Spacer(1, 18),
            Paragraph(
                "This report is provided for personal informational and research-pilot purposes only. "
                "It does not diagnose, treat, or predict any medical condition. Scores are sample "
                "feature indices and should not be interpreted as clinical measurements.",
                styles["Italic"],
            ),
        ]
    )
    document.build(story)
    return buffer.getvalue()
