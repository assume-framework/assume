# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime

try:
    import torch as th
except ImportError:
    th = None

from assume.common.fast_pandas.index import FastIndex
from assume.common.fast_pandas.series import FastSeries


class TensorFastSeries(FastSeries):
    """
    A specialized version of FastSeries designed to handle tensors.
    """

    def __init__(self, index: FastIndex, value=None, name: str = ""):
        """
        Initialize a TensorFastSeries.

        Parameters:
            index (FastIndex): The index for the series.
            value (torch.Tensor | float | None): The initial value to populate the series.
                If a scalar (e.g., 0.0) is provided, it will be converted to a tensor.
                Defaults to None.
            name (str, optional): The name of the series. Defaults to "".
        """
        super().__init__(index=index, value=None, name=name)

        # Ensure _data is initialized to hold tensors
        if value is None:
            self._data = [None for _ in range(len(index))]
        elif isinstance(value, th.Tensor):
            self._data = [value.clone() for _ in range(len(index))]
        elif isinstance(value, (int | float)):
            self._data = [th.tensor(value) for _ in range(len(index))]
        else:
            raise TypeError(
                f"Unsupported value type: {type(value)}. Must be torch.Tensor, float, or int."
            )

    def __setitem__(self, item: int | datetime | slice, value):
        """
        Assign tensor value(s) to item(s) in the series.

        Parameters:
            item (int | datetime | slice): The index or slice.
            value (th.Tensor): The tensor value(s) to assign.
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._data):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._data)}"
                )
            self._data[item] = value.clone()
        elif isinstance(item, slice):
            start_idx = item.start or 0
            stop_idx = item.stop or len(self._data)
            step = item.step or 1
            slice_length = len(range(start_idx, stop_idx, step))

            if len(value) != slice_length:
                raise ValueError(
                    f"Length of value ({len(value)}) does not match the length of the slice ({slice_length})."
                )
            for i, idx in enumerate(range(start_idx, stop_idx, step)):
                self._data[idx] = value[i].clone()
        elif isinstance(item, datetime):
            idx = self.index.get_position(item)
            self._data[idx] = value.clone()
        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be int, slice, or datetime."
            )

    def __getitem__(self, item: int | datetime | slice):
        """
        Retrieve tensor(s) from the series.

        Parameters:
            item (int | datetime | slice): The index or slice.

        Returns:
            th.Tensor | list[th.Tensor]: The retrieved tensor(s).
        """
        if isinstance(item, int):
            if item < 0 or item >= len(self._data):
                raise IndexError(
                    f"Index {item} is out of bounds for series of length {len(self._data)}"
                )
            return self._data[item]
        elif isinstance(item, slice):
            start_idx = item.start or 0
            stop_idx = item.stop or len(self._data)
            step = item.step or 1
            return [self._data[i] for i in range(start_idx, stop_idx, step)]
        elif isinstance(item, datetime):
            idx = self.index.get_position(item)
            return self._data[idx]
        else:
            raise TypeError(
                f"Unsupported index type: {type(item)}. Must be int, slice, or datetime."
            )

    def copy(self, deep: bool = False):
        """
        Create a copy of the TensorFastSeries.

        Parameters:
            deep (bool): If True, perform a deep copy. Defaults to False.

        Returns:
            TensorFastSeries: A new instance with copied data.
        """
        if deep:
            copied_data = [
                tensor.clone() if tensor is not None else None for tensor in self._data
            ]
        else:
            copied_data = self._data[:]

        return TensorFastSeries(
            index=self._index,
            value=None,  # We'll manually set _data below
            name=self._name,
        )._set_data(copied_data)

    def _set_data(self, data):
        """
        Helper method to set data during initialization.

        Parameters:
            data (list[th.Tensor]): The data to set.

        Returns:
            TensorFastSeries: The modified instance.
        """
        self._data = data
        return self

    def __repr__(self) -> str:
        """
        Return a string representation of the TensorFastSeries.

        Returns:
            str: A string describing the series.
        """
        preview_length = 3  # Number of items to preview from the start and end
        total_length = len(self._data)

        if total_length == 0:
            return f"TensorFastSeries(name='{self._name}', length=0, data=[])"

        # Preview a subset of the data
        start_preview = self._data[:preview_length]
        end_preview = (
            self._data[-preview_length:] if total_length > preview_length else []
        )

        preview = (
            start_preview
            + (["..."] if total_length > 2 * preview_length else [])
            + end_preview
        )

        return (
            f"TensorFastSeries(name='{self._name}', length={total_length}, "
            f"data={preview})"
        )

    def __str__(self) -> str:
        """
        Informal string representation of the FastSeries, identical to __repr__.

        Returns:
            str: String representation of the FastSeries.
        """
        return self.__repr__()
