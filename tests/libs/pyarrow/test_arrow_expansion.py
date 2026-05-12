import pytest
import pyarrow as pa
from dlt.common.schema import Schema
from dlt.normalize.items_normalizers.arrow import _flatten_py_arrow_item
from dlt.common.normalizers.naming import NamingConvention
from dlt.common.normalizers.json.helpers import TJsonColumnExpansionSpec

@pytest.fixture
def naming() -> NamingConvention:
    return Schema("default").naming

def test_ac1_basic_arrow_flatten(naming: NamingConvention) -> None:
    """AC1: Arrow StructArray is expanded into __ sub-columns."""
    struct_type = pa.struct([
        pa.field("name", pa.string()),
        pa.field("email", pa.string())
    ])
    table = pa.table({
        "id": [1],
        "metadata": pa.array([{"name": "John", "email": "john@example.com"}], type=struct_type)
    })
    
    flattened = _flatten_py_arrow_item(table, naming)
    
    assert "id" in flattened.column_names
    assert "metadata__name" in flattened.column_names
    assert "metadata__email" in flattened.column_names
    assert "metadata" not in flattened.column_names
    assert flattened["metadata__name"].to_pylist() == ["John"]
    assert flattened["metadata__email"].to_pylist() == ["john@example.com"]

def test_ac2_keep_original_not_supported(naming: NamingConvention) -> None:
    """AC2: Current Arrow flattener drops the original struct column (no keep_original yet)."""
    struct_type = pa.struct([pa.field("f1", pa.int32())])
    table = pa.table({
        "data": pa.array([{"f1": 10}], type=struct_type)
    })
    
    flattened = _flatten_py_arrow_item(table, naming)
    assert "data__f1" in flattened.column_names
    assert "data" not in flattened.column_names

def test_ac3_json_string_expansion(naming: NamingConvention) -> None:
    """AC3: JSON string column is parsed and expanded."""
    table = pa.table({
        "id": [1],
        "data": pa.array(['{"user": {"name": "John", "age": 30}}'], type=pa.string())
    })
    
    expansion_cols = {
        "data": TJsonColumnExpansionSpec(flatten_spec=True, keep_original=False, force_string=False, max_depth=None)
    }
    
    flattened = _flatten_py_arrow_item(table, naming, json_expansion_cols=expansion_cols)
    
    assert "data__user__name" in flattened.column_names
    assert "data__user__age" in flattened.column_names
    assert flattened["data__user__name"].to_pylist() == ["John"]
    assert flattened["data__user__age"].to_pylist() == [30]

def test_ac4_max_depth_equivalent(naming: NamingConvention) -> None:
    """AC4: max_nesting depth controls how many levels of structs are expanded."""
    inner = pa.struct([pa.field("val", pa.int32())])
    outer = pa.struct([pa.field("child", inner)])
    table = pa.table({
        "data": pa.array([{"child": {"val": 42}}], type=outer)
    })
    
    # Depth 1: expands 'data' -> 'data__child' (keeps child as struct)
    flattened = _flatten_py_arrow_item(table, naming, max_nesting=1)
    assert "data__child" in flattened.column_names
    assert pa.types.is_struct(flattened.schema.field("data__child").type)
    
    # Depth 2: expands 'data' -> 'data__child' -> 'data__child__val'
    flattened = _flatten_py_arrow_item(table, naming, max_nesting=2)
    assert "data__child__val" in flattened.column_names
    assert flattened["data__child__val"].to_pylist() == [42]

def test_ac5_is_nested_type_protection(naming: NamingConvention) -> None:
    """AC5: Columns identified as nested types are preserved as StructArrays."""
    struct_type = pa.struct([pa.field("f1", pa.int32())])
    table = pa.table({
        "normal": pa.array([{"f1": 1}], type=struct_type),
        "protected": pa.array([{"f1": 2}], type=struct_type)
    })
    
    def is_nested_type(name, lvl):
        return name == "protected"
        
    flattened = _flatten_py_arrow_item(table, naming, is_nested_type=is_nested_type)
    
    assert "normal__f1" in flattened.column_names
    assert "protected" in flattened.column_names
    assert pa.types.is_struct(flattened.schema.field("protected").type)

def test_chunked_streaming_parity(naming: NamingConvention) -> None:
    """Ensure consistency when handling ChunkedArrays (Streaming data)."""
    struct_type = pa.struct([pa.field("val", pa.int32())])
    c1 = pa.array([{"val": 1}], type=struct_type)
    c2 = pa.array([{"val": 2}], type=struct_type)
    
    table = pa.table({
        "data": pa.chunked_array([c1, c2])
    })
    
    flattened = _flatten_py_arrow_item(table, naming)
    assert flattened["data__val"].to_pylist() == [1, 2]
    assert flattened["data__val"].num_chunks == 2
