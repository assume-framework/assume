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
    """Returns a factory that produces the DataFrame you ask for."""

    def _make(table: str) -> pd.DataFrame:
        if table == "rl_params":
            return pd.DataFrame(
                {
                    "dt": ["2024-01-01", "2024-01-02"],
                    "unit": ["pp_6", "pp_7"],
                    "profit": [100, 200],
                    "reward": [0.5, 0.7],
                    "regret": [0.1, 0.2],
                    "noise_0": [0.1, 0.1],
                    "noise_1": [0.2, 0.2],
                }
            )
        elif table == "rl_grad_params":
            return pd.DataFrame(
                {
                    "step": [1, 2],
                    "unit": ["pp_6", "pp_7"],
                    "actor_loss": [0.1, 0.2],
                    "actor_total_grad_norm": [3, 5],
                    "actor_max_grad_norm": [5, 7],
                    "critic_loss": [0.2, 0.4],
                    "critic_total_grad_norm": [4, 6],
                    "critic_max_grad_norm": [6, 8],
                    "lr": [0.001, 0.001],
                }
            )
        else:
            raise ValueError(f"Unknown table '{table}'")

    return _make


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
        pd.DataFrame({"name": sample_df("rl_params").columns.tolist()}),
        sample_df("rl_params"),
        sample_df("rl_grad_params"),
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
        "02_train/01_episode_reward",
        "02_train/02_reward",
        "02_train/03_profit",
        "02_train/04_regret",
        "02_train/05_noise",
        "03_grad/06_learning_rate",
        "03_grad/07_actor_loss",
        "03_grad/08_actor_total_grad_norm",
        "03_grad/09_actor_max_grad_norm",
        "03_grad/10_critic_loss",
        "03_grad/11_critic_total_grad_norm",
        "03_grad/12_critic_max_grad_norm",
    ]
    assert all(metric in metrics_logged for metric in expected_metrics)
