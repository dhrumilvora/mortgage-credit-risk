from credit_risk.pipelines.reporting import run_reporting_pipeline


def test_run_reporting_pipeline_can_be_skipped():
    config = {
        "parameters": {
            "reporting": {
                "skip": True,
            }
        }
    }

    assert run_reporting_pipeline(config) is None
