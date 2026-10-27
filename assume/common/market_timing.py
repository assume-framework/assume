# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Visualization and sanity checks of the timing of market configs.

Each market opening consists of a bidding window (opening until clearing) and
the delivery period of the traded products. Agents submit their bids as soon as
a market opens, so a market can only use results of markets which were cleared
strictly before its opening.
"""

from datetime import datetime, timedelta
from typing import NamedTuple

import termaid

from assume.common.market_objects import MarketConfig
from assume.common.utils import get_available_products


class MarketOpening(NamedTuple):
    """
    A single opening of a market.

    Args:
        market_id (str): the ID of the market
        opening (datetime.datetime): when the market opens and agents submit bids
        closing (datetime.datetime): when the market is cleared and results are sent
        delivery_start (datetime.datetime): start of the earliest product
        delivery_end (datetime.datetime): end of the latest product
        product_count (int): number of products traded in this opening
    """

    market_id: str
    opening: datetime
    closing: datetime
    delivery_start: datetime
    delivery_end: datetime
    product_count: int


class TimingIssue(NamedTuple):
    """
    A problem found in the timing of the market configs.

    Args:
        level (str): "error" if the setup is invalid, "warning" if it is likely unintended
        market_id (str): the ID of the affected market
        opening (datetime.datetime): the affected opening of the market
        message (str): description of the issue
        clash_start (datetime.datetime): start of the period in which the timings clash
        clash_end (datetime.datetime): end of the period in which the timings clash
        label (str): short description of the issue to label it in a diagram
    """

    level: str
    market_id: str
    opening: datetime
    message: str
    clash_start: datetime
    clash_end: datetime
    label: str


def get_market_openings(
    market_config: MarketConfig, start: datetime, end: datetime
) -> list[MarketOpening]:
    """
    Get all openings of a market which open in the interval [start, end).

    Args:
        market_config (MarketConfig): the market config
        start (datetime.datetime): start of the interval
        end (datetime.datetime): end of the interval (exclusive)

    Returns:
        list[MarketOpening]: the openings of the market
    """
    until = market_config.opening_hours._until
    openings = []
    for opening in market_config.opening_hours.xafter(start, inc=True):
        if opening >= end:
            break
        closing = opening + market_config.opening_duration
        if until and closing > until:
            # the market does not open, as the clearing is after the markets end time
            break
        products = get_available_products(market_config.market_products, opening)
        openings.append(
            MarketOpening(
                market_id=market_config.market_id,
                opening=opening,
                closing=closing,
                delivery_start=min(product[0] for product in products),
                delivery_end=max(product[1] for product in products),
                product_count=len(products),
            )
        )
    return openings


def get_default_window(market_configs: list[MarketConfig]) -> tuple[datetime, datetime]:
    """
    Get a window which contains one full cycle of all markets.

    The window starts at the earliest market opening and ends one opening period of the
    slowest market after the latest first market opening.

    Args:
        market_configs (list[MarketConfig]): the market configs

    Returns:
        tuple[datetime.datetime, datetime.datetime]: start and end of the window
    """
    first_openings = [config.opening_hours[0] for config in market_configs]
    periods = []
    for config, first_opening in zip(market_configs, first_openings):
        second_opening = config.opening_hours.after(first_opening)
        if second_opening is None:
            periods.append(config.opening_duration)
        else:
            periods.append(second_opening - first_opening)
    return min(first_openings), max(first_openings) + max(periods)


def check_market_timing(
    market_configs: list[MarketConfig],
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[TimingIssue]:
    """
    Check the timing of the given market configs for common mistakes.

    The following is checked for all openings in [start, end):

    - error: the bidding window closes after the first product is delivered
    - warning: a market opens at the same time another market is cleared and both trade
      an overlapping delivery period, so the results of that clearing are not yet
      available to the bidders of the opening market
    - warning: a market opens again before its previous opening was cleared

    Args:
        market_configs (list[MarketConfig]): the market configs
        start (datetime.datetime | None): start of the checked interval, defaults to the first market opening
        end (datetime.datetime | None): end of the checked interval, defaults to one full cycle of all markets

    Returns:
        list[TimingIssue]: the issues found, empty if the timing is fine
    """
    default_start, default_end = get_default_window(market_configs)
    start = start or default_start
    end = end or default_end
    openings = {
        config.market_id: get_market_openings(config, start, end)
        for config in market_configs
    }

    issues = []
    for market_id, market_openings in openings.items():
        for market_opening in market_openings:
            if market_opening.closing > market_opening.delivery_start:
                issues.append(
                    TimingIssue(
                        "error",
                        market_id,
                        market_opening.opening,
                        f"{market_id} opening at {market_opening.opening} closes at "
                        f"{market_opening.closing}, after its delivery starts at "
                        f"{market_opening.delivery_start}",
                        market_opening.delivery_start,
                        market_opening.closing,
                        f"{market_id} bids during delivery",
                    )
                )

        for previous, current in zip(market_openings, market_openings[1:]):
            if current.opening < previous.closing:
                issues.append(
                    TimingIssue(
                        "warning",
                        market_id,
                        current.opening,
                        f"{market_id} opens at {current.opening} before its previous "
                        f"opening is cleared at {previous.closing}",
                        current.opening,
                        previous.closing,
                        f"{market_id} openings overlap",
                    )
                )

    for market_id, market_openings in openings.items():
        for other_id, other_openings in openings.items():
            if other_id == market_id:
                continue
            for market_opening in market_openings:
                for other in other_openings:
                    # only relevant if the other market clears the same delivery period
                    delivery_overlaps = (
                        market_opening.delivery_start < other.delivery_end
                        and other.delivery_start < market_opening.delivery_end
                    )
                    if market_opening.opening != other.closing or not delivery_overlaps:
                        continue
                    issues.append(
                        TimingIssue(
                            "warning",
                            market_id,
                            market_opening.opening,
                            f"{market_id} opens at {market_opening.opening} when "
                            f"{other_id} is cleared for an overlapping delivery period, "
                            f"so these {other_id} results are not available to bidders "
                            f"on {market_id}",
                            market_opening.opening,
                            market_opening.opening,
                            f"{market_id} opens as {other_id} clears",
                        )
                    )

    return sorted(issues, key=lambda issue: (issue.opening, issue.market_id))


def _mermaid_name(name: str) -> str:
    # colons and hashes have a special meaning in the mermaid gantt syntax
    return name.replace(":", "_").replace("#", "_")


def _mermaid_duration(duration: timedelta) -> str:
    return f"{int(duration.total_seconds() // 60)}m"


def market_timing_to_mermaid(
    market_configs: list[MarketConfig],
    start: datetime | None = None,
    end: datetime | None = None,
    title: str = "Market timing",
) -> str:
    """
    Create a mermaid gantt diagram of the market openings in [start, end).

    For every opening, the bidding window, the clearing and the delivery period are shown.
    Openings affected by a timing issue are marked as critical and prefixed with "!".
    The periods in which the timings clash are shown in an additional section.

    Args:
        market_configs (list[MarketConfig]): the market configs
        start (datetime.datetime | None): start of the shown interval, defaults to the first market opening
        end (datetime.datetime | None): end of the shown interval, defaults to one full cycle of all markets
        title (str): title of the diagram

    Returns:
        str: the mermaid diagram
    """
    default_start, default_end = get_default_window(market_configs)
    start = start or default_start
    end = end or default_end
    timing_issues = check_market_timing(market_configs, start, end)
    issues = {(issue.market_id, issue.opening) for issue in timing_issues}
    date_format = "%Y-%m-%d %H:%M"

    lines = [
        "gantt",
        f"    title {_mermaid_name(title)}",
        "    dateFormat YYYY-MM-DD HH:mm",
        "    axisFormat %d %H:%M",
        "    todayMarker off",
    ]
    for config in market_configs:
        lines.append(f"    section {_mermaid_name(config.market_id)}")
        for i, market_opening in enumerate(
            get_market_openings(config, start, end), start=1
        ):
            affected = (config.market_id, market_opening.opening) in issues
            tag = "crit, " if affected else ""
            prefix = "! " if affected else ""
            bidding_duration = market_opening.closing - market_opening.opening
            delivery_duration = (
                market_opening.delivery_end - market_opening.delivery_start
            )
            products = "product" if market_opening.product_count == 1 else "products"
            lines += [
                f"    {prefix}{i} bidding :{tag}{market_opening.opening:{date_format}}, "
                f"{_mermaid_duration(bidding_duration)}",
                f"    {prefix}{i} clearing :milestone, {tag}{market_opening.closing:{date_format}}, 0m",
                f"    {i} delivery {market_opening.product_count} {products} :active, "
                f"{market_opening.delivery_start:{date_format}}, "
                f"{_mermaid_duration(delivery_duration)}",
            ]

    if timing_issues:
        lines.append("    section timing clashes")
    for issue in timing_issues:
        clash_duration = issue.clash_end - issue.clash_start
        tag = "crit" if clash_duration else "milestone, crit"
        # colons can not be used in task names, so the time is written as 04h00
        lines.append(
            f"    {issue.clash_start:%d %Hh%M} {_mermaid_name(issue.label)} :{tag}, "
            f"{issue.clash_start:{date_format}}, {_mermaid_duration(clash_duration)}"
        )
    return "\n".join(lines)


def render_market_timing(
    market_configs: list[MarketConfig],
    start: datetime | None = None,
    end: datetime | None = None,
    title: str = "Market timing",
    use_ascii: bool = False,
) -> str:
    """
    Render the market timing as text for the terminal using termaid, followed by a legend
    and the timing issues. The start of each timing clash is marked by a vertical line.

    Args:
        market_configs (list[MarketConfig]): the market configs
        start (datetime.datetime | None): start of the shown interval, defaults to the first market opening
        end (datetime.datetime | None): end of the shown interval, defaults to one full cycle of all markets
        title (str): title of the diagram
        use_ascii (bool): use ASCII instead of unicode box drawing characters

    Returns:
        str: the rendered diagram and the timing issues
    """
    mermaid = market_timing_to_mermaid(market_configs, start, end, title)
    issues = check_market_timing(market_configs, start, end)

    # vertical markers are a termaid extension of the gantt syntax,
    # so they are only added for the terminal and not to the mermaid diagram
    lines = mermaid.split("\n")
    clash_times = sorted({issue.clash_start for issue in issues})
    markers = [f"    vert {clash_time:%Y-%m-%d %H:%M}" for clash_time in clash_times]
    text = termaid.render(
        "\n".join(lines[:5] + markers + lines[5:]), use_ascii=use_ascii
    )
    if markers:
        # termaid labels the markers in the row below the title, where close markers
        # overlap, the clash times are part of the labels of the clash section instead
        text_lines = text.split("\n")
        text_lines[1] = ""
        text = "\n".join(text_lines)

    if use_ascii:
        legend = "Legend: # bidding  * clearing  = delivery  ! timing issue  | clash"
    else:
        legend = "Legend: █ bidding  ◆ clearing  ▓ delivery  ! timing issue  ┊ clash"

    if not issues:
        return f"{text}\n\n{legend}\n\nNo timing issues found."
    issue_lines = [f"{issue.level.upper()}: {issue.message}" for issue in issues]
    return "\n".join([text, "", legend, ""] + issue_lines)
