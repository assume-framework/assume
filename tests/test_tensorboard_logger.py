# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import Mock, patch

import pandas as pd
import pytest

try:
    from assume.reinforcement_learning.tensorboard_logger import TensorBoardLogger
except ImportError:
    pass


@pytest.fixture
def mock_db():
    """Fixture for mocking database connection"""
    with patch("sqlalchemy.create_engine") as mock_engine:
        # Create a mock engine of type sqlite
        engine = Mock()
        engine.dialect.name = "sqlite"
        mock_engine.return_value = engine
        yield engine


@pytest.fixture
def mock_writer():
    """Fixture for mocking SummaryWriter"""
    with patch("torch.utils.tensorboard.SummaryWriter") as mock_writer:
        writer_instance = Mock()
        mock_writer.return_value = writer_instance
        yield writer_instance


@pytest.fixture
def sample_df():
    """Fixture for creating sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "dt": ["2024-01-01", "2024-01-02"],
            "unit": ["pp_6", "pp_7"],
            "profit": [100, 200],
            "reward": [0.5, 0.7],
            "regret": [0.1, 0.2],
            "loss": [0.01, 0.02],
            "total_grad_norm": [3, 5],
            "max_grad_norm": [5, 7],
            "lr": [0.001, 0.001],
            "noise_0": [0.1, 0.1],
            "noise_1": [0.2, 0.2],
        }
    )


@pytest.mark.require_learning
def test_initialization():
    """Test basic initialization of TensorBoardLogger"""
    logger = TensorBoardLogger(
        db_uri="sqlite:///:memory:",
        simulation_id="sim_1",
        learning_mode=True,
    )

    assert logger.simulation_id == "sim_1"
    assert logger.learning_mode is True
    assert logger.evaluation_mode is False
    assert logger.episodes_collecting_initial_experience == 0
    assert logger.episode == 1


@patch("pandas.read_sql")
@pytest.mark.require_learning
def test_update_tensorboard_training_mode(
    mock_read_sql, mock_db, mock_writer, sample_df
):
    """Test update_tensorboard method in training mode"""
    # Setup
    mock_read_sql.side_effect = [
        pd.DataFrame({"name": sample_df.columns.tolist()}),
        sample_df,
    ]

    logger = TensorBoardLogger(
        db_uri="sqlite:///:memory:",
        simulation_id="sim_1",
        learning_mode=True,
    )
    logger.db = mock_db
    logger.writer = mock_writer

    # Execute
    logger.update_tensorboard()

    # Verify
    assert mock_writer.add_scalar.called
    # Verify specific metrics were logged
    calls = mock_writer.add_scalar.call_args_list
    metrics_logged = [call[0][0] for call in calls]
    expected_metrics = [
        "train/01_episode_reward",
        "train/02_reward",
        "train/03_profit",
        "train/04_regret",
        "train/05_learning_rate",
        "train/06_loss",
        "train/07_total_grad_norm",
        "train/08_max_grad_norm",
        "train/09_noise",
    ]
    assert all(metric in metrics_logged for metric in expected_metrics)
