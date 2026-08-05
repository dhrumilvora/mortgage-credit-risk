from credit_risk.data.schemas import PERFORMANCE_SCHEMA


def test_performance_schema_has_35_fields():
    assert len(PERFORMANCE_SCHEMA) == 35


def test_performance_positions_are_contiguous():
    positions = [field.position for field in PERFORMANCE_SCHEMA]

    assert positions == list(range(1, 36))


def test_performance_field_names_are_unique():
    names = [field.name for field in PERFORMANCE_SCHEMA]

    assert len(names) == len(set(names))


def test_loan_id_is_first_field():
    assert PERFORMANCE_SCHEMA[0].name == "loan_id"
