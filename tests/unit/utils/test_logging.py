import logging

from credit_risk.utils.logging import ConsoleFormatter, configure_logging


def test_configure_logging_can_disable_credit_risk_logs():
    configure_logging(enabled=False)

    assert logging.getLogger("credit_risk").disabled

    configure_logging(enabled=True)


def test_console_formatter_shortens_credit_risk_component_name():
    record = logging.LogRecord(
        name="credit_risk.pipelines.ingest",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Ingestion started",
        args=(),
        exc_info=None,
    )

    result = ConsoleFormatter(use_colors=False).format(record)

    assert "ingest" in result
    assert "Ingestion started" in result
