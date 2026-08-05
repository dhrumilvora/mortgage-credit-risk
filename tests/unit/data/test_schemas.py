from credit_risk.data.schemas import PERFORMANCE_SCHEMA, ORIGINATION_SCHEMA


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


def test_origination_schema_has_31_fields():
    assert len(ORIGINATION_SCHEMA) == 31


def test_origination_positions_are_contiguous():
    positions = [field.position for field in ORIGINATION_SCHEMA]

    assert positions == list(range(1, 32))


def test_origination_field_names_are_unique():
    names = [field.name for field in ORIGINATION_SCHEMA]

    assert len(names) == len(set(names))


def test_origination_loan_id_position():
    assert ORIGINATION_SCHEMA[19].name == "loan_id"
