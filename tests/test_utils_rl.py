# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from assume.common.utils import convert_tensors


def test_convert_orderbook_list_with_numpy_floats():
    """Test conversion of orderbook list with numpy floats (real data structure)."""

    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Simulate real orderbook structure
    orderbook = [
        {
            "start_time": datetime(2019, 3, 1, 1, 0),
            "end_time": datetime(2019, 3, 1, 2, 0),
            "only_hours": None,
            # these torch.float32 as in the repository in general
            "price": th.tensor(3000.0),
            "volume": th.tensor(-4920.7),
            "node": "node0",
            "bid_id": "demand_1",
            "unit_id": "demand",
        },
        {
            "start_time": datetime(2019, 3, 1, 1, 0),
            "end_time": datetime(2019, 3, 1, 2, 0),
            "only_hours": None,
            "price": np.float64(12.5),
            "volume": np.float64(1000.0),
            "node": "node0",
            "bid_id": "pp_1",
            "unit_id": "pp_1",
        },
    ]

    result = convert_tensors(orderbook)

    # Check structure is preserved
    assert len(result) == 2
    # We convert to floats, so we check for approximate equality
    # we chose relative & absolute tolerance here because the values are quite different in magnitude
    assert result[0]["price"] == pytest.approx(3000.0, rel=1e-6, abs=1e-9)
    assert result[0]["volume"] == pytest.approx(-4920.7, rel=1e-6, abs=1e-9)
    assert result[1]["price"] == 12.5
    assert result[1]["volume"] == 1000.0
    assert result[0]["start_time"] == datetime(2019, 3, 1, 1, 0)


def test_convert_rl_params_list_with_numpy_floats():
    """Test conversion of RL params list with numpy floats. This is also a list of dicts so it should work similar as before"""

    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    rl_data = [
        {
            "datetime": datetime(2019, 3, 1, 1, 0),
            "unit": "pp_10",
            "reward": np.float64(-0.00039),
            "regret": 0.0,
            "profit": np.float64(-19.55),
            # these torch.float32 as in the repository in general
            "actions_0": th.tensor(0.336),
            "actions_1": th.tensor(0.446),
        }
    ]

    result = convert_tensors(rl_data)

    assert len(result) == 1
    assert result[0]["reward"] == -0.00039
    assert result[0]["profit"] == -19.55
    assert result[0]["unit"] == "pp_10"
    # We convert to floats, so we check for approximate equality
    # we chose relative & absolute tolerance here because the values are quite small
    assert result[0]["actions_0"] == pytest.approx(0.336, rel=1e-6, abs=1e-9)
    assert result[0]["actions_1"] == pytest.approx(0.446, rel=1e-6, abs=1e-9)


def test_convert_single_tensor():
    """Test conversion of a single PyTorch tensor."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create a simple tensor
    tensor = th.tensor([1.0, 2.0, 3.0])
    result = convert_tensors(tensor)

    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]


def test_convert_series_with_tensors():
    """Test conversion of pandas Series containing tensors."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create a Series with tensors
    data = {
        "a": th.tensor([1.0, 2.0]),
        "b": th.tensor([3.0, 4.0]),
        "c": th.tensor([5.0, 6.0]),
    }
    series = pd.Series(data)
    result = convert_tensors(series)

    # Check that it's still a Series
    assert isinstance(result, pd.Series)

    # Check order is preserved
    assert list(result.index) == ["a", "b", "c"]

    # Check values are converted correctly
    assert result["a"] == [1.0, 2.0]
    assert result["b"] == [3.0, 4.0]
    assert result["c"] == [5.0, 6.0]


def test_convert_list_of_dicts_with_tensors():
    """Test conversion of list of dictionaries containing tensors."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create a list of dicts with tensors
    data = [
        {"id": 1, "tensor": th.tensor([1.0, 2.0]), "name": "first"},
        {"id": 2, "tensor": th.tensor([3.0, 4.0]), "name": "second"},
        {"id": 3, "tensor": th.tensor([5.0, 6.0]), "name": "third"},
    ]
    result = convert_tensors(data)

    # Check list length and order
    assert len(result) == 3
    assert result[0]["name"] == "first"
    assert result[1]["name"] == "second"
    assert result[2]["name"] == "third"

    # Check tensor values are converted
    assert result[0]["tensor"] == [1.0, 2.0]
    assert result[1]["tensor"] == [3.0, 4.0]
    assert result[2]["tensor"] == [5.0, 6.0]

    # Check non-tensor values are unchanged
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
    assert result[2]["id"] == 3


def test_convert_nested_dict_with_tensors():
    """Test conversion of nested dictionaries with tensors."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create nested dict with tensors
    data = {
        "level1": {
            "tensor_a": th.tensor([1.0, 2.0]),
            "level2": {
                "tensor_b": th.tensor([3.0, 4.0]),
                "scalar": 42,
            },
        },
        "simple_tensor": th.tensor([5.0, 6.0]),
    }
    result = convert_tensors(data)

    # Check nested structure is preserved
    assert result["level1"]["tensor_a"] == [1.0, 2.0]
    assert result["level1"]["level2"]["tensor_b"] == [3.0, 4.0]
    assert result["level1"]["level2"]["scalar"] == 42
    assert result["simple_tensor"] == [5.0, 6.0]


def test_convert_mixed_data_types():
    """Test conversion of mixed data types (tensors and non-tensors)."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    data = {
        "tensor": th.tensor([1.0, 2.0, 3.0]),
        "list": [1, 2, 3],
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "none": None,
    }
    result = convert_tensors(data)

    # Check all values are preserved
    assert result["tensor"] == [1.0, 2.0, 3.0]
    assert result["list"] == [1, 2, 3]
    assert result["string"] == "hello"
    assert result["int"] == 42
    assert result["float"] == 3.14
    assert result["none"] is None


def test_convert_multidimensional_tensor():
    """Test conversion of multi-dimensional tensors."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create 2D tensor
    tensor_2d = th.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = convert_tensors(tensor_2d)

    assert isinstance(result, list)
    assert result == [[1.0, 2.0], [3.0, 4.0]]


def test_convert_series_order_preservation():
    """Test that Series order is strictly preserved during conversion."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create ordered series with specific indices
    indices = ["z", "a", "m", "b", "x"]
    data = {idx: th.tensor([float(i), float(i + 1)]) for i, idx in enumerate(indices)}
    series = pd.Series(data, index=indices)

    result = convert_tensors(series)

    # Check order matches original
    assert list(result.index) == indices
    for i, idx in enumerate(indices):
        assert result[idx] == [float(i), float(i + 1)]


def test_convert_empty_structures():
    """Test conversion of empty structures."""

    # Empty dict
    assert convert_tensors({}) == {}

    # Empty list
    assert convert_tensors([]) == []

    # Empty Series
    result = convert_tensors(pd.Series(dtype=object))
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_convert_no_tensors_present():
    """Test that data without tensors is returned unchanged."""
    data = {
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "string": "test",
        "number": 42.5,
    }
    result = convert_tensors(data)

    # Should return the same data structure
    assert result == data


def test_convert_series_with_numeric_values():
    """Test that numeric Series with tensors maintains correct values."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Test with different numeric types
    data = {
        "int_tensor": th.tensor([1, 2, 3]),
        "float_tensor": th.tensor([1.5, 2.5, 3.5]),
        "mixed": th.tensor([1.0, 2, 3.0]),
    }
    series = pd.Series(data)
    result = convert_tensors(series)

    assert result["int_tensor"] == [1, 2, 3]
    assert result["float_tensor"] == [1.5, 2.5, 3.5]
    assert result["mixed"] == [1.0, 2.0, 3.0]


def test_convert_large_tensor():
    """Test conversion of larger tensors to ensure efficiency."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create a larger tensor
    large_tensor = th.randn(1000, 50)
    result = convert_tensors(large_tensor)

    assert isinstance(result, list)
    assert len(result) == 1000
    assert len(result[0]) == 50


def test_convert_dataframe_with_tensor_columns():
    """Test conversion of DataFrame with columns containing tensors."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    # Create DataFrame with mixed column types
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["unit_a", "unit_b", "unit_c"],
            "tensor_values": [
                th.tensor([1.0, 2.0]),
                th.tensor([3.0, 4.0]),
                th.tensor([5.0, 6.0]),
            ],
            "scalar_tensor": [th.tensor(10.5), th.tensor(20.5), th.tensor(30.5)],
            "normal_value": [100, 200, 300],
        }
    )

    result = pd.DataFrame()
    for col in df.columns:
        result[col] = df[col].apply(convert_tensors)

    # Check that result is still a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that DataFrame structure is preserved
    assert len(result) == 3
    assert list(result.columns) == [
        "id",
        "name",
        "tensor_values",
        "scalar_tensor",
        "normal_value",
    ]

    # Check tensor columns are converted to lists
    assert result["tensor_values"].iloc[0] == pytest.approx(
        [1.0, 2.0], rel=1e-6, abs=1e-9
    )
    assert result["tensor_values"].iloc[1] == pytest.approx(
        [3.0, 4.0], rel=1e-6, abs=1e-9
    )
    assert result["tensor_values"].iloc[2] == pytest.approx(
        [5.0, 6.0], rel=1e-6, abs=1e-9
    )

    # Check scalar tensors are converted
    assert result["scalar_tensor"].iloc[0] == pytest.approx(10.5, rel=1e-6, abs=1e-9)
    assert result["scalar_tensor"].iloc[1] == pytest.approx(20.5, rel=1e-6, abs=1e-9)
    assert result["scalar_tensor"].iloc[2] == pytest.approx(30.5, rel=1e-6, abs=1e-9)

    # Check non-tensor columns remain unchanged
    assert result["id"].tolist() == [1, 2, 3]
    assert result["name"].tolist() == ["unit_a", "unit_b", "unit_c"]
    assert result["normal_value"].tolist() == [100, 200, 300]


def test_transform_buffer_data():
    """Test transform_buffer_data converts nested dict to numpy array correctly."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    from assume.reinforcement_learning.learning_utils import transform_buffer_data

    # Create test data: {datetime -> {unit_id -> [tensor]}}
    data = {
        datetime(2019, 3, 1, 1, 0): {
            "unit_a": [th.tensor([1.0, 2.0])],
            "unit_b": [th.tensor([3.0, 4.0])],
        },
        datetime(2019, 3, 1, 2, 0): {
            "unit_a": [th.tensor([5.0, 6.0])],
            "unit_b": [th.tensor([7.0, 8.0])],
        },
        datetime(2019, 3, 1, 3, 0): {
            "unit_a": [th.tensor([9.0, 10.0])],
            "unit_b": [th.tensor([11.0, 12.0])],
        },
    }

    result = transform_buffer_data(data, device=th.device("cpu"))

    # Check output is numpy array
    assert isinstance(result, np.ndarray)

    # Check shape: (n_timesteps=3, n_units=2, feature_dim=2)
    assert result.shape == (3, 2, 2)

    # Check values are correct (sorted by timestamp and unit_id)
    # unit_a comes before unit_b alphabetically
    assert result[0, 0] == pytest.approx([1.0, 2.0], rel=1e-6, abs=1e-9)  # t=0, unit_a
    assert result[0, 1] == pytest.approx([3.0, 4.0], rel=1e-6, abs=1e-9)  # t=0, unit_b
    assert result[1, 0] == pytest.approx([5.0, 6.0], rel=1e-6, abs=1e-9)  # t=1, unit_a
    assert result[1, 1] == pytest.approx([7.0, 8.0], rel=1e-6, abs=1e-9)  # t=1, unit_b
    assert result[2, 0] == pytest.approx([9.0, 10.0], rel=1e-6, abs=1e-9)  # t=2, unit_a
    assert result[2, 1] == pytest.approx(
        [11.0, 12.0], rel=1e-6, abs=1e-9
    )  # t=2, unit_b


def test_transform_buffer_data_scalar_values():
    """Test transform_buffer_data with scalar values (rewards)."""
    try:
        import torch as th
    except ImportError:
        pytest.skip("PyTorch not installed")

    from assume.reinforcement_learning.learning_utils import transform_buffer_data

    # Test with scalar values (like rewards)
    data = {
        datetime(2019, 3, 1, 1, 0): {
            "unit_1": [th.tensor(0.5)],
            "unit_2": [th.tensor(0.8)],
        },
        datetime(2019, 3, 1, 2, 0): {
            "unit_1": [th.tensor(-0.3)],
            "unit_2": [th.tensor(1.2)],
        },
    }

    result = transform_buffer_data(data, device=th.device("cpu"))

    # Check shape: (n_timesteps=2, n_units=2, feature_dim=1)
    assert result.shape == (2, 2, 1)

    # Check scalar values
    assert result[0, 0, 0] == pytest.approx(0.5, rel=1e-6, abs=1e-9)
    assert result[0, 1, 0] == pytest.approx(0.8, rel=1e-6, abs=1e-9)
    assert result[1, 0, 0] == pytest.approx(-0.3, rel=1e-6, abs=1e-9)
    assert result[1, 1, 0] == pytest.approx(1.2, rel=1e-6, abs=1e-9)


def test_convert_without_torch_installed():
    """Test that convert_tensors handles missing PyTorch gracefully."""
    # This test simulates PyTorch not being installed

    data = {
        "list": [1, 2, 3],
        "string": "test",
        "number": 42.5,
    }

    result = convert_tensors(data)

    # Should return data unchanged if torch is not available
    assert result == data
