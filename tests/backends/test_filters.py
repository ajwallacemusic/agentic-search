from agentic_search.backends.filters import filter_fields, matches
from agentic_search.core.types import And, Contains, Eq, Exists, In, Not, Or, Range

REC = {"type": "drug", "year": 2020, "tags": ["a", "b"], "date": "2024-03-01",
       "nested": {"x": 5}, "none": None}


def test_eq_scalar_and_list():
    assert matches(Eq(field="type", value="drug"), REC)
    assert not matches(Eq(field="type", value="history"), REC)
    assert matches(Eq(field="tags", value="a"), REC)


def test_in():
    assert matches(In(field="year", values=[2019, 2020]), REC)
    assert matches(In(field="tags", values=["z", "b"]), REC)
    assert not matches(In(field="year", values=[1]), REC)


def test_range_numbers_and_iso_dates():
    assert matches(Range(field="year", gte=2020, lt=2021), REC)
    assert not matches(Range(field="year", gt=2020), REC)
    assert matches(Range(field="date", gte="2024-01-01"), REC)


def test_range_type_mismatch_is_false():
    assert not matches(Range(field="type", gte=3), REC)


def test_exists_and_missing():
    assert matches(Exists(field="type"), REC)
    assert not matches(Exists(field="none"), REC)
    assert not matches(Exists(field="missing"), REC)
    assert not matches(Eq(field="missing", value=1), REC)


def test_contains_case_insensitive():
    assert matches(Contains(field="type", value="DR"), REC)
    assert matches(Contains(field="tags", value="b"), REC)


def test_dotted_path():
    assert matches(Eq(field="nested.x", value=5), REC)


def test_boolean_combinators():
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Eq(field="year", value=1), Not(clause=Eq(field="year", value=2))])])
    assert matches(f, REC)


def test_filter_fields():
    f = And(clauses=[Eq(field="a", value=1), Not(clause=Or(clauses=[Exists(field="b")]))])
    assert filter_fields(f) == {"a", "b"}
