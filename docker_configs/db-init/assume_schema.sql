-- SPDX-FileCopyrightText: ASSUME Developers
--
-- SPDX-License-Identifier: AGPL-3.0-or-later

-- 0) Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis;

-- 1) market_meta (partitioned by simulation)
CREATE TABLE IF NOT EXISTS market_meta (
  simulation               TEXT      NOT NULL,
  market_id                TEXT,
  "index"                  INTEGER,
  time                     TIMESTAMP NOT NULL,
  node                     TEXT,
  product_start            TIMESTAMP NOT NULL,
  product_end              TIMESTAMP NOT NULL,
  only_hours               TEXT,
  price                    REAL,
  max_price                REAL,
  min_price                REAL,
  supply_volume            REAL,
  supply_volume_energy     REAL,
  demand_volume            REAL,
  demand_volume_energy     REAL
)
PARTITION BY LIST (simulation);

-- 2) market_dispatch (partitioned by simulation)
CREATE TABLE IF NOT EXISTS market_dispatch (
  simulation  TEXT    NOT NULL,
  "index"     INTEGER,
  market_id   TEXT,
  datetime    TIMESTAMP NOT NULL,
  unit_id     TEXT,
  power       REAL
)
PARTITION BY LIST (simulation);

-- 3) market_orders (static)
CREATE TABLE IF NOT EXISTS market_orders (
  simulation           TEXT      NOT NULL,
  market_id            TEXT,
  start_time           TIMESTAMP NOT NULL,
  volume               REAL,
  accepted_volume      REAL,
  price                REAL,
  unit_id              TEXT,
  bid_type             TEXT,
  node                 TEXT,
  evaluation_frequency TEXT,
  eligible_lambda      TEXT
);

-- 4) unit_dispatch (partitioned by simulation)
CREATE TABLE IF NOT EXISTS unit_dispatch (
  simulation               TEXT    NOT NULL,
  time                     TIMESTAMP NOT NULL,
  "index"                  INTEGER,
  unit                     TEXT,
  power                    REAL,
  heat                     REAL,
  soc                      REAL,
  energy_generation_costs  REAL,
  energy_cashflow          REAL,
  total_costs              REAL
)
PARTITION BY LIST (simulation);

-- 5.1) power_plant_meta (static)
CREATE TABLE IF NOT EXISTS power_plant_meta (
  simulation      TEXT    NOT NULL,
  "index"         TEXT,
  technology      TEXT,
  unit_operator   TEXT,
  node            TEXT,
  max_power       REAL,
  min_power       REAL,
  emission_factor REAL,
  efficiency      REAL
);

-- 5.2) storage_meta (static)
CREATE TABLE IF NOT EXISTS storage_meta (
  simulation            TEXT    NOT NULL,
  "index"               TEXT,
  unit_type             TEXT,
  max_soc               REAL,
  min_soc               REAL,
  max_power_charge      REAL,
  max_power_discharge   REAL,
  min_power_charge      REAL,
  min_power_discharge   REAL,
  efficiency_charge     REAL,
  efficiency_discharge  REAL
);

-- 5.3) demand_meta (static; add additional fields as needed)
CREATE TABLE IF NOT EXISTS demand_meta (
  simulation  TEXT    NOT NULL,
  "index"     TEXT,
  unit_type   TEXT,
  max_power   REAL,
  min_power   REAL,
);

-- 5.4) exchange_meta (static; add additional fields as needed)
CREATE TABLE IF NOT EXISTS exchange_meta (
  simulation   TEXT    NOT NULL,
  "index"      TEXT,
  unit_type    TEXT,
  price_import REAL,
  price_export REAL,
);

-- 6) rl_params (partitioned by simulation)
CREATE TABLE IF NOT EXISTS rl_params (
  simulation         TEXT      NOT NULL,
  "index"            INTEGER,
  unit               TEXT,
  datetime           TIMESTAMP NOT NULL,
  evaluation_mode    BOOLEAN,
  episode            INTEGER,
  profit             REAL,
  reward             REAL,
  regret             REAL,
  actions            TEXT,
  exploration_noise  REAL,
  critic_loss        REAL,
  total_grad_norm    REAL,
  max_grad_norm      REAL,
  learning_rate      REAL
)
PARTITION BY LIST (simulation);

-- 7) rl_meta (static)
CREATE TABLE IF NOT EXISTS rl_meta (
  simulation       TEXT    NOT NULL,
  "index"          INTEGER,
  episode          INTEGER,
  eval_episode     INTEGER,
  learning_mode    BOOLEAN,
  evaluation_mode  BOOLEAN
);

-- 8) grid_flows (partitioned by simulation)
CREATE TABLE IF NOT EXISTS grid_flows (
  "index"     INTEGER,
  datetime    TIMESTAMP NOT NULL,
  line        TEXT,
  flow        REAL,
  simulation  TEXT
)
PARTITION BY LIST (simulation);

-- 9) kpis (partitioned by simulation)
CREATE TABLE IF NOT EXISTS kpis (
  "index"     INTEGER,
  variable    TEXT,
  ident       TEXT,
  value       REAL,
  simulation  TEXT,
  time        TIMESTAMP DEFAULT now()
)
PARTITION BY LIST (simulation);
