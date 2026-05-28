"""Unit tests for Arrow-native struct flattening functions."""

import pytest

import json as _json

import pytest

from dlt.common.libs.pyarrow import (
    MAX_RECURSION_DEPTH,
    _resolve_casefold_collisions,
    apply_arrow_depth_limit,
    apply_arrow_force_string,
    apply_arrow_path_filter,
    flatten_struct_column,
    is_empty_struct,
    is_flattenable_column,
    pyarrow as pa,
)
from dlt.common.normalizers.naming import NamingConvention
from dlt.common.normalizers.naming.snake_case import NamingConvention as SnakeCaseNamingConvention


@pytest.fixture
def naming() -> SnakeCaseNamingConvention:
    return SnakeCaseNamingConvention()


def test_flatten_basic_struct(naming: NamingConvention) -> None:
    """struct<name: string, age: int64> decomposes into data__name, data__age."""
    tbl = pa.table({"id": [1], "data": [{"name": "John", "age": 30}]})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert "data" not in result.column_names
    assert result.column("data__name")[0].as_py() == "John"
    assert result.column("data__age")[0].as_py() == 30
    assert types["data__name"] == pa.string()
    assert types["data__age"] in (pa.int32(), pa.int64())


def test_flatten_nested_struct(naming: NamingConvention) -> None:
    """struct<a: struct<b: int32>> with r_lvl=100 → parent__a__b."""
    inner = pa.struct([pa.field("b", pa.int32())])
    outer = pa.struct([pa.field("a", inner)])
    tbl = pa.table({"data": pa.array([{"a": {"b": 42}}], type=outer)})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 100)
    assert "data__a__b" in result.column_names
    assert result.column("data__a__b")[0].as_py() == 42


def test_flatten_max_nesting_exhausted(naming: NamingConvention) -> None:
    """_r_lvl=0 → nested struct kept as struct column, not recursed into."""
    inner = pa.struct([pa.field("b", pa.int32())])
    outer = pa.struct([pa.field("a", inner)])
    tbl = pa.table({"data": pa.array([{"a": {"b": 42}}], type=outer)})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 0)
    # data__a exists as a struct (not flattened to data__a__b)
    assert "data__a__b" not in result.column_names
    assert "data__a" in result.column_names
    assert pa.types.is_struct(result.schema.field("data__a").type)
    assert "data" not in result.column_names


def test_flatten_null_struct(naming: NamingConvention) -> None:
    """Null parent row → null child columns."""
    stype = pa.struct([("name", pa.string()), ("age", pa.int32())])
    tbl = pa.table({"data": pa.array([None, {"name": "Jane", "age": 25}], type=stype)})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert result.column("data__name")[0].as_py() is None
    assert result.column("data__name")[1].as_py() == "Jane"


def test_flatten_empty_struct(naming: NamingConvention) -> None:
    """struct<> → column dropped, no new columns produced."""
    tbl = pa.table({"id": [1], "data": pa.array([{}], type=pa.struct([]))})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert "data" not in result.column_names
    assert "id" in result.column_names
    assert len(types) == 0


def test_flatten_preserves_non_struct_columns(naming: NamingConvention) -> None:
    """Non-struct columns are untouched after flattening."""
    tbl = pa.table({"id": [1], "x": [10], "data": [{"name": "John"}]})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert "id" in result.column_names
    assert "x" in result.column_names
    assert result.column("id")[0].as_py() == 1
    assert result.column("x")[0].as_py() == 10


def test_flatten_record_batch(naming: NamingConvention) -> None:
    """Same flattening behavior when input table was built from RecordBatch chunks."""
    tbl = pa.table({"id": [1], "data": [{"name": "John", "age": 30}]})
    batch = tbl.to_batches()[0]
    tbl_from_batch = pa.Table.from_batches([batch])
    result, types = flatten_struct_column(tbl_from_batch, "data", naming, {}, 1000)
    assert "data__name" in result.column_names
    assert result.column("data__name")[0].as_py() == "John"


def test_path_filter_single_field(naming: NamingConvention) -> None:
    """x-json-flatten: ["user.name"] — only that field path is retained."""
    inner = pa.struct([pa.field("name", pa.string()), pa.field("email", pa.string())])
    stype = pa.struct([pa.field("a", pa.int32()), pa.field("user", inner)])
    arr = pa.array([{"a": 1, "user": {"name": "John", "email": "j@e.com"}}], type=stype)
    result = apply_arrow_path_filter(arr, ["user.name"])
    assert result.type.num_fields == 1
    assert result.type.field(0).name == "user"


def test_keep_original_struct(naming: NamingConvention) -> None:
    """Flatten drops the original struct column by default (no keep-original)."""
    tbl = pa.table({"id": [1], "data": [{"name": "John"}]})
    result, types = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert "data" not in result.column_names
    assert "data__name" in result.column_names


def test_force_string_scalars() -> None:
    """force_string — int leaf coerced to pa.string(), preserving nulls."""
    stype = pa.struct([pa.field("x", pa.int32()), pa.field("y", pa.string())])
    arr = pa.array([{"x": 42, "y": "hello"}], type=stype)
    result = apply_arrow_force_string(arr)
    assert pa.types.is_string(result.type.field(0).type)
    assert result.field("x")[0].as_py() == "42"


def test_max_depth_1() -> None:
    """max_depth=1 — nested struct children serialized to JSON string, root shape preserved."""
    import json as _json

    inner = pa.struct([pa.field("x", pa.int32())])
    stype = pa.struct([pa.field("nested", inner)])
    arr = pa.array([{"nested": {"x": 42}}], type=stype)
    result = apply_arrow_depth_limit(arr, 1)
    # root struct preserved; nested struct child becomes JSON string
    assert pa.types.is_struct(result.type)
    assert result.type.field(0).type == pa.string()
    assert _json.loads(result.field("nested")[0].as_py()) == {"x": 42}


def test_non_struct_with_hint_skipped() -> None:
    """int64 with x-json-flatten hint — is_flattenable_column returns False."""
    assert is_flattenable_column(pa.int64(), "bad", {"x-json-flatten": True}) is False


def test_explicit_false_hint() -> None:
    """x-json-flatten: False — struct is not flattenable."""
    st = pa.struct([pa.field("x", pa.int32())])
    assert is_flattenable_column(st, "data", {"x-json-flatten": False}) is False


def test_empty_struct_guard() -> None:
    """is_empty_struct returns True only for zero-field structs."""
    assert is_empty_struct(pa.struct([])) is True
    assert is_empty_struct(pa.struct([pa.field("x", pa.int32())])) is False


def test_max_recursion_depth_constant() -> None:
    """MAX_RECURSION_DEPTH is 100."""
    assert MAX_RECURSION_DEPTH == 100


def test_idempotent_second_pass(naming: NamingConvention) -> None:
    """Flattening the same source twice produces identical column sets."""
    tbl = pa.table({"id": [1], "data": [{"name": "John"}]})
    result1, _ = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert "data" not in result1.column_names
    assert "data__name" in result1.column_names
    result2, _ = flatten_struct_column(tbl, "data", naming, {}, 1000)
    assert result1.column_names == result2.column_names


def test_flatten_multiple_struct_columns(naming: NamingConvention) -> None:
    """Two struct columns flattened sequentially — each expanded independently."""
    tbl = pa.table({"a": [{"x": 1}], "b": [{"y": 2}]})
    result1, _ = flatten_struct_column(tbl, "a", naming, {}, 1000)
    assert "a__x" in result1.column_names
    assert "b" in result1.column_names
    assert "a" not in result1.column_names
    result2, _ = flatten_struct_column(result1, "b", naming, {}, 1000)
    assert "a__x" in result2.column_names
    assert "b__y" in result2.column_names
    assert "a" not in result2.column_names
    assert "b" not in result2.column_names


def test_apply_arrow_depth_limit_zero() -> None:
    """max_depth=0 collapses root struct to a pa.string() array."""
    st = pa.struct([pa.field("x", pa.int32()), pa.field("y", pa.string())])
    arr = pa.array([{"x": 1, "y": "hello"}, None], type=st)
    result = apply_arrow_depth_limit(arr, 0)
    assert pa.types.is_string(result.type)
    assert result[0].as_py() == '{"x":1,"y":"hello"}'
    assert result[1].as_py() is None


def test_casefold_collision_suffix(naming: NamingConvention) -> None:
    """Fields differing only by case get __c_<hash> suffixes under casefold."""
    names = {"data__tuan": pa.array([1]), "data__Tuan": pa.array([2])}
    fields = {n: pa.field(n, pa.int32()) for n in names}
    resolved, _ = _resolve_casefold_collisions(names, fields, "data", naming, str.casefold)
    result_names = list(resolved.keys())
    assert len(result_names) == 2
    assert all("__c_" in n for n in result_names)


def test_casefold_no_collision_no_suffix(naming: NamingConvention) -> None:
    """Fields without case collision keep names under casefold."""
    names = {"data__name": pa.array(["a"]), "data__age": pa.array([1])}
    fields = {n: pa.field(n, pa.int32()) for n in names}
    resolved, _ = _resolve_casefold_collisions(names, fields, "data", naming, str.casefold)
    assert "data__name" in resolved
    assert "data__age" in resolved
