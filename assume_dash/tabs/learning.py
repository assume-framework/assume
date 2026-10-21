# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Learning tab: reinforcement learning training progress (Grafana "Training progress")."""

import re

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

from assume_dash.components.controls import SIMULATION
from assume_dash.components.widgets import (
    empty_figure,
    graph,
    row,
    section,
    style_figure,
)
from assume_dash.tabs.base import Tab

MODES = {"training": False, "evaluation": True}


class LearningTab(Tab):
    id = "learning"
    title = "Learning"

    def available(self) -> bool:
        return self.data.has("rl_params")

    def layout(self):
        return html.Div(
            [
                section(
                    "All units",
                    row(
                        graph(self.cid("avg-reward-training")),
                        graph(self.cid("avg-reward-evaluation")),
                    ),
                ),
                section(
                    "Per unit",
                    html.Label("RL units"),
                    dcc.Dropdown(id=self.cid("units"), multi=True),
                    *[
                        row(
                            graph(self.cid(f"{metric}-total-training")),
                            graph(self.cid(f"{metric}-total-evaluation")),
                        )
                        for metric in ("reward", "profit")
                    ],
                ),
                section(
                    "Within an episode",
                    row(
                        [
                            html.Label("Training episode"),
                            dcc.Dropdown(
                                id=self.cid("episode-training"), clearable=False
                            ),
                        ],
                        [
                            html.Label("Evaluation episode"),
                            dcc.Dropdown(
                                id=self.cid("episode-evaluation"), clearable=False
                            ),
                        ],
                    ),
                    *[
                        row(
                            graph(self.cid(f"{metric}-episode-training")),
                            graph(self.cid(f"{metric}-episode-evaluation")),
                        )
                        for metric in ("reward", "profit", "actions")
                    ],
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        units = Input(self.cid("units"), "value")

        @app.callback(
            Output(self.cid("units"), "options"),
            Output(self.cid("units"), "value"),
            Output(self.cid("episode-training"), "options"),
            Output(self.cid("episode-training"), "value"),
            Output(self.cid("episode-evaluation"), "options"),
            Output(self.cid("episode-evaluation"), "value"),
            Input(SIMULATION, "value"),
        )
        def options(simulation):
            df = self._rl(simulation)
            if df.empty:
                return [], [], [], None, [], None
            unit_ids = sorted(df["unit"].unique())
            episodes = {
                mode: sorted(df.loc[df["eval"] == flag, "episode"].unique())
                for mode, flag in MODES.items()
            }
            # the last episode is the most interesting one
            return (
                unit_ids,
                unit_ids,
                episodes["training"],
                episodes["training"][-1] if episodes["training"] else None,
                episodes["evaluation"],
                episodes["evaluation"][-1] if episodes["evaluation"] else None,
            )

        outputs = [Output(self.cid(f"avg-reward-{m}"), "figure") for m in MODES] + [
            Output(self.cid(f"{metric}-total-{m}"), "figure")
            for metric in ("reward", "profit")
            for m in MODES
        ]

        @app.callback(*outputs, Input(SIMULATION, "value"), units)
        def totals(simulation, selected):
            df = self._rl(simulation)
            figs = []
            for mode, flag in MODES.items():
                figs.append(self._average_reward(df, flag, mode))
            for metric in ("reward", "profit"):
                for mode, flag in MODES.items():
                    figs.append(self._total_per_unit(df, flag, metric, selected, mode))
            return figs

        episode_outputs = [
            Output(self.cid(f"{metric}-episode-{m}"), "figure")
            for metric in ("reward", "profit", "actions")
            for m in MODES
        ]

        @app.callback(
            *episode_outputs,
            Input(SIMULATION, "value"),
            units,
            Input(self.cid("episode-training"), "value"),
            Input(self.cid("episode-evaluation"), "value"),
        )
        def episodes(simulation, selected, train_episode, eval_episode):
            df = self._rl(simulation)
            chosen = {"training": train_episode, "evaluation": eval_episode}
            figs = []
            for metric in ("reward", "profit", "actions"):
                for mode, flag in MODES.items():
                    sub = self._episode(df, flag, chosen[mode], selected)
                    figs.append(self._episode_figure(sub, metric, mode, chosen[mode]))
            return figs

    def _rl(self, simulation: str | None) -> pd.DataFrame:
        df = self.data.table("rl_params", simulation)
        if df.empty:
            return df
        flag = df["evaluation_mode"].astype(str).str.lower().isin(["true", "1"])
        return df.assign(eval=flag)

    @staticmethod
    def _episode(df, flag, episode, selected) -> pd.DataFrame:
        if df.empty or episode is None:
            return pd.DataFrame()
        mask = (df["eval"] == flag) & (df["episode"] == episode)
        return df[mask & df["unit"].isin(selected or [])]

    @staticmethod
    def _average_reward(df: pd.DataFrame, flag: bool, mode: str):
        sub = df[df["eval"] == flag] if not df.empty else df
        if sub.empty:
            return empty_figure(f"No {mode} data")
        per_unit = sub.groupby(["episode", "unit"])["reward"].sum()
        avg = per_unit.groupby("episode").mean().reset_index()
        fig = px.line(avg, x="episode", y="reward", markers=True)
        return style_figure(fig, f"Average {mode} reward", "Reward")

    @staticmethod
    def _total_per_unit(df, flag, metric, selected, mode):
        sub = (
            df[(df["eval"] == flag) & df["unit"].isin(selected or [])]
            if not df.empty
            else df
        )
        if sub.empty or metric not in sub:
            return empty_figure(f"No {mode} data")
        total = sub.groupby(["episode", "unit"], as_index=False)[metric].sum()
        fig = px.line(total, x="episode", y=metric, color="unit", markers=True)
        return style_figure(fig, f"Total {mode} {metric}", metric.capitalize())

    @staticmethod
    def _episode_figure(df, metric, mode, episode):
        if df.empty:
            return empty_figure(f"No {mode} data")
        title = f"{mode.capitalize()} {metric} during episode {episode}"
        if metric == "actions":
            return LearningTab._actions_figure(df, title)
        df = df.assign(profit=df["profit"] / 1000) if metric == "profit" else df
        fig = px.line(df.sort_values("datetime"), x="datetime", y=metric, color="unit")
        return style_figure(fig, title, "kEUR" if metric == "profit" else "Reward")

    @staticmethod
    def _actions_figure(df: pd.DataFrame, title: str):
        action_cols = sorted(
            (c for c in df.columns if re.fullmatch(r"actions_\d+", c)),
            key=lambda c: int(c.split("_")[1]),
        )
        frames = []
        for col in action_cols:
            idx = col.split("_")[1]
            noise = f"exploration_noise_{idx}"
            part = df[["datetime", "unit"]].assign(
                action=idx, value=df[col], kind="noisy"
            )
            frames.append(part)
            if noise in df.columns and df[noise].notna().any():
                frames.append(part.assign(value=df[col] - df[noise], kind="noise-free"))
        if not frames:
            return empty_figure("No actions recorded")
        long = pd.concat(frames).dropna(subset=["value"]).sort_values("datetime")
        long["series"] = "action " + long["action"] + " (" + long["kind"] + ")"
        fig = px.line(long, x="datetime", y="value", color="unit", line_dash="series")
        return style_figure(fig, title)
