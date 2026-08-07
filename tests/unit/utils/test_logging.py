import logging

from credit_risk.utils.logging import configure_logging


def test_configure_logging_can_disable_credit_risk_logs():
    configure_logging(enabled=False)

    assert logging.getLogger("credit_risk").disabled

    configure_logging(enabled=True)

