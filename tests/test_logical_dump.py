from graphrag_drift.dump_utils import quote_identifier


def test_quote_identifier_keeps_safe_names():
    assert quote_identifier("Entity") == "Entity"
    assert quote_identifier("RELATED_TO") == "RELATED_TO"


def test_quote_identifier_escapes_dynamic_identifiers():
    assert quote_identifier("Has Space") == "`Has Space`"
    assert quote_identifier("a`b") == "`a``b`"
