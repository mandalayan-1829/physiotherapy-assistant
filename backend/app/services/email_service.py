"""Email delivery abstraction.

Two transports are provided:

* ``SmtpEmailSender`` — real delivery through any SMTP provider (configured
  entirely from environment variables; no credentials are ever hardcoded).
  Works with any provider, so switching vendors never touches the password-reset
  logic.
* ``ConsoleEmailSender`` — development fallback used when SMTP is not
  configured. It records the message in memory for tests and logs an explicit
  warning that **no email was actually sent**.

Message bodies are treated as sensitive: they contain single-use password reset
codes, so **bodies are never logged** by any transport, in any environment. The
in-memory ``outbox`` of the console transport is the only place a body is kept,
and it exists solely so the automated tests can complete the flow. Production
refuses to start without a real mail transport (see ``app.core.config``).

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
        use_ssl: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls
        self.timeout = timeout
        # Implicit TLS (SMTPS, usually port 465) versus STARTTLS (usually 587).
        self.use_ssl = use_ssl or port == 465

    def send(self, message: OutboundEmail) -> DeliveryResult:
        email = EmailMessage()
        email["From"] = formataddr((self.from_name, self.from_email))
        email["To"] = message.to
        email["Subject"] = message.subject
        email.set_content(message.text_body)
        if message.html_body:
            email.add_alternative(message.html_body, subtype="html")

        try:
            if self.use_ssl:
                server_ctx = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout)
            else:
                server_ctx = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
            with server_ctx as server:
                if self.use_tls and not self.use_ssl:
                    server.starttls()
                if self.username:
                    server.login(self.username, self.password)
                server.send_message(email)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            # Only the transport-level error is logged. The message body (which
            # carries the reset code) is never included.
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
    """Development transport: records the message in memory for tests.

    The body is **never logged**. A reset code that reaches a log file has
    effectively leaked: logs are copied, shipped to aggregators and read by
    people who should not be able to reset a patient's password. Tests read the
    code from :func:`get_console_outbox` instead.
    """

    transport = "console"
    outbox: list[OutboundEmail] = field(default_factory=list)

    def send(self, message: OutboundEmail) -> DeliveryResult:
        self.outbox.append(message)
        # Only the fact that delivery was attempted is logged - not the subject,
        # the recipient or the body. The body carries the reset code, and even the
        # subject is unnecessary detail to retain.
        logger.warning(
            "EMAIL NOT SENT (console backend). Configure SMTP_HOST/SMTP_USERNAME/"
            "SMTP_PASSWORD or set EMAIL_BACKEND=smtp to deliver real mail. "
            "Message contents (including any verification code) are withheld from logs."
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
    """Build the configured sender. SMTP settings are read from the environment.

    Any standards-compliant provider works (Resend, Brevo, SendGrid, Postmark,
    AWS SES, a corporate relay, …) because delivery is plain SMTP. Port 465 is
    treated as implicit TLS, everything else uses STARTTLS when enabled.
    """
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
            use_ssl=settings.smtp_use_ssl,
        )
    return _console_sender


def get_console_outbox() -> list[OutboundEmail]:
    """Expose recorded messages.

    Test-only: this is the one place a message body is retained, so the suite can
    complete a password reset without the code ever being logged.
    """
    return _console_sender.outbox


def clear_console_outbox() -> None:
    _console_sender.outbox.clear()
