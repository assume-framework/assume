.. SPDX-FileCopyrightText: ASSUME Developers
..
.. SPDX-License-Identifier: MIT

.. _conventions:

Conventions and Definitions
===========================

This page collects the conventions that are used throughout ASSUME. Most of them are explained in more detail
on the respective pages, which are linked in each section. If you write your own units, bidding strategies or
market clearing algorithms, please follow these conventions so that your components work together with the
existing ones.

Time
----

**Naive datetimes are UTC.**
All datetimes in ASSUME are naive (they carry no timezone information) and are interpreted as UTC.
This applies to the ``start_date`` and ``end_date`` in the config, to the time index of all input time series,
to market products and to the outputs.

Python and mango do not follow this convention by default. ``datetime.timestamp()`` and
``datetime.fromtimestamp()`` interpret naive datetimes in the local timezone of the machine, and since mango 2.3.0
the same holds for recurrence rules with a naive ``dtstart``. Using them directly leads to results that silently
differ between machines in different timezones (e.g. a laptop and a server). Therefore, use the helpers from
:mod:`assume.common.utils` instead:

 ====================================================================== ================================================================
  instead of                                                             use
 ====================================================================== ================================================================
  ``dt.timestamp()``                                                     :func:`~assume.common.utils.datetime2timestamp`
  ``datetime.fromtimestamp(ts)``                                         :func:`~assume.common.utils.timestamp2datetime`
  ``rr.rrule(freq, dtstart=start, until=end, ...)``                      :func:`~assume.common.utils.utc_rrule` (same arguments)
  ``rr.rrule`` from a frequency string such as ``"1h"``                  :func:`~assume.common.utils.create_rrule`
 ====================================================================== ================================================================

.. note::
   To catch violations of this convention, the CI runs the tests with the machine timezone set to
   ``Pacific/Chatham`` (UTC+12:45/+13:45). Code that uses the local timezone by accident fails there instead of passing
   silently on machines that run in UTC.

**Timestamps label the start of a time step.**
A value at timestamp ``t`` describes the time step from ``t`` to ``t + freq``, where ``freq`` is the time
resolution of the simulation. For example, a dispatch of 50 MW at 22:00 with an hourly resolution means that
50 MW are delivered from 22:00 to 23:00. Market products are defined by a ``start`` and an ``end`` and cover
the interval from ``start`` (included) to ``end`` (excluded), see :doc:`market_config`.

**State of charge.**
In contrast to dispatch values, a state of charge (SoC) is a state at a point in time and not a value over a time
step. ``SoC[t]`` is the state of charge at the beginning of time step ``t``, before the dispatch of that time step.
The dispatch in ``t`` therefore only changes ``SoC[t+1]``. See :doc:`storages` for details.

Units of Measurement
--------------------

- Power and volumes of bids are given in MW, energy in MWh and prices in €/MWh (or the currency of your input data).
- The state of charge is given relative to the capacity of the storage and is always between 0 and 1.

Signs
-----

ASSUME uses signed values to represent the direction of power flows and trades. The following table summarises the
conventions. The pages linked in the last column explain them in more detail.

 ============================================ ================================================== ============================================== =========================
  quantity                                     positive                                           negative                                       details
 ============================================ ================================================== ============================================== =========================
  trade volume (orders and accepted volumes)   sold to the market (e.g. generation)               bought from the market (e.g. demand)           :doc:`market_config`
  storage power / volume                       discharging                                        charging                                       :doc:`storages`
  redispatch volume                            upward redispatch                                  downward redispatch                            :doc:`redispatch_modeling`
 ============================================ ================================================== ============================================== =========================

Prices can be positive or negative, and their sign does not depend on the direction of the trade. Together with
the sign of the volume, the sign of the price determines the direction of the financial flow, see :doc:`market_config`.

The ramping parameters follow the sign of the power they belong to:

 ==================================================== =========== =========================
  parameter                                            sign        details
 ==================================================== =========== =========================
  ``ramp_up`` / ``ramp_down`` (power plants)            ``>= 0``    :doc:`powerplants`
  ``ramp_up_discharge`` / ``ramp_down_discharge``       ``>= 0``    :doc:`storages`
  ``ramp_up_charge`` / ``ramp_down_charge``             ``<= 0``    :doc:`storages`
 ==================================================== =========== =========================

For power plants, a ramp of ``0`` or no value means that the ramping is not limited.
Since the charging power of a storage is negative, "ramping up the charge" means that the charging power becomes more
negative. In the same way, ``max_power_charge`` is the most negative charging power and therefore smaller than or equal to
``min_power_charge``.
