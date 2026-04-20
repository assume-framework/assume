# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest

try:
    from assume.common.grid_utils import get_supported_solver_linopy
except ImportError:
    pass

@pytest.mark.require_network
def test_solver_available():
    assert get_supported_solver_linopy() == "highs"
    assert get_supported_solver_linopy("unknown_solver") == "highs"
