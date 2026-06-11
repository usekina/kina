"""Email delivery helpers for KinaBot V1 verification codes."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from config import (
    SMTP_FROM_EMAIL,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_USE_TLS,
    VERIFICATION_CODE_TTL_MINUTES,
)


def email_delivery_configured() -> bool:
    return bool(SMTP_HOST and SMTP_PORT and SMTP_FROM_EMAIL)


def send_verification_email(to_email: str, code: str) -> tuple[bool, str]:
    if not email_delivery_configured():
        return False, "Email delivery is not configured."

    message = EmailMessage()
    message["Subject"] = "Your KinaBot verification code"
    message["From"] = SMTP_FROM_EMAIL
    message["To"] = to_email
    message.set_content(
        "\n".join(
            [
                "Your KinaBot verification code is:",
                "",
                code,
                "",
                f"This code expires in {VERIFICATION_CODE_TTL_MINUTES} minutes.",
                "",
                "If you did not request this code, you can ignore this email.",
            ]
        )
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
    except Exception as exc:
        return False, f"Email delivery failed: {exc}"

    return True, "Verification code sent by email."
