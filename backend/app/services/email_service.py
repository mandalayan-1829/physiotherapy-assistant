"""Email delivery abstraction.

Two transports are provided:

* ``SmtpEmailSender`` — real delivery through any SMTP provider (configured
  entirely from environment variables; no credentials are ever hardcoded).
* ``ConsoleEmailSender`` — development fallback used when SMTP is not
  configured. It records the message server-side and logs an explicit warning
  that **no email was actually sent**. It never reports success.

Delivery results are returned to the caller so that an unconfigured or failing
mail server is never silently presented to the user as a sent email.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.utils import formataddr

from app.core.config import settings

logger = logging.getLogger("physioai.email")


@dataclass
class OutboundEmail:
    to: str
    subject: str
    text_body: str
    html_body: str | None = None


@dataclass
class DeliveryResult:
    delivered: bool
    transport: str
    error: str | None = None
    detail: str = ""


class EmailSender:
    transport = "base"

    def send(self, message: OutboundEmail) -> DeliveryResult:  # pragma: no cover - interface
        raise NotImplementedError


class SmtpEmailSender(EmailSender):
    transport = "smtp"

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_email: str,
        from_name: str,
        use_tls: bool,
        timeout: int,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls
        self.timeout = timeout

    def send(self, message: OutboundEmail) -> DeliveryResult:
        email = EmailMessage()
        email["From"] = formataddr((self.from_name, self.from_email))
        email["To"] = message.to
        email["Subject"] = message.subject
        email.set_content(message.text_body)
        if message.html_body:
            email.add_alternative(message.html_body, subtype="html")

        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(email)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            logger.error("SMTP delivery to %s failed: %s", message.to, exc)
            return DeliveryResult(
                delivered=False,
                transport=self.transport,
                error=str(exc),
                detail="The mail server rejected the message.",
            )

        return DeliveryResult(
            delivered=True, transport=self.transport, detail="Message accepted by the mail server."
        )


@dataclass
class ConsoleEmailSender(EmailSender):
    """Development transport: records the message for local inspection."""

    transport = "console"
    outbox: list[OutboundEmail] = field(default_factory=list)

    def send(self, message: OutboundEmail) -> DeliveryResult:
        self.outbox.append(message)
        logger.warning(
            "EMAIL NOT SENT (console backend). Configure SMTP_HOST/SMTP_USERNAME/"
            "SMTP_PASSWORD or set EMAIL_BACKEND=smtp to deliver real mail.\n"
            "Recipient: %s\nSubject: %s\n--- development-only message body ---\n%s\n"
            "-------------------------------------",
            message.to,
            message.subject,
            # The body (and therefore the verification code) is printed only by
            # this development transport. The SMTP transport never logs the
            # message contents, and the console transport is only selected when
            # no mail provider is configured.
            message.text_body,
        )
        return DeliveryResult(
            delivered=False,
            transport=self.transport,
            detail=(
                "No mail transport is configured; the message was recorded server-side "
                "and was NOT delivered."
            ),
        )


_console_sender = ConsoleEmailSender()


def get_email_sender() -> EmailSender:
    """Build the configured sender. SMTP settings are read from the environment."""
    if settings.resolved_email_backend() == "smtp":
        return SmtpEmailSender(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            from_email=settings.smtp_from_email,
            from_name=settings.smtp_from_name,
            use_tls=settings.smtp_use_tls,
            timeout=settings.email_timeout_seconds,
        )
    return _console_sender


def get_console_outbox() -> list[OutboundEmail]:
    """Expose recorded messages. Used by tests and local development only."""
    return _console_sender.outbox


def clear_console_outbox() -> None:
    _console_sender.outbox.clear()
