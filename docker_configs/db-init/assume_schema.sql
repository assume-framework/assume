-- SPDX-FileCopyrightText: ASSUME Developers
-- SPDX-License-Identifier: AGPL-3.0-or-later

-- 0) Enable extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis;

----------------------------
-- 1) market_meta (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS market_meta (
  simulation               TEXT      NOT NULL,
  market_id                TEXT,
  time                     TIMESTAMP NOT NULL,
  product_start            TIMESTAMP NOT NULL,
  product_end              TIMESTAMP NOT NULL,
  only_hours               TIMESTAMP,
  supply_volume            REAL,
  demand_volume            REAL,
  supply_volume_energy     REAL,
  demand_volume_energy     REAL,
  price                    REAL,
  max_price                REAL,
  min_price                REAL,
  node                     TEXT,
  PRIMARY KEY (simulation, market_id, node, time)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON market_meta (simulation, time);
CREATE INDEX idx_market_meta_sim_market_prod_start
  ON market_meta (simulation, market_id, product_start);

----------------------------
-- 2) market_dispatch (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS market_dispatch (
  simulation  TEXT    NOT NULL,
  market_id   TEXT,
  datetime    TIMESTAMP NOT NULL,
  unit_id     TEXT,
  power       REAL,
  PRIMARY KEY (simulation, market_id, datetime, unit_id)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON market_dispatch (simulation, datetime);
CREATE INDEX idx_market_dispatch_sim_unit_datetime
  ON market_dispatch (simulation, unit_id, datetime);

----------------------------
-- 3) market_orders (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS market_orders (
  simulation           TEXT      NOT NULL,
  market_id            TEXT,
  start_time           TIMESTAMP NOT NULL,
  end_time             TIMESTAMP NOT NULL,
  price                REAL,
  volume               REAL,
  bid_type             TEXT,
  node                 TEXT,
  bid_id               TEXT,
  unit_id              TEXT,
  accepted_price       REAL,
  accepted_volume      REAL,
  parent_bid_id        TEXT,
  min_acceptance_ratio REAL,
  PRIMARY KEY (simulation, market_id, start_time, bid_id)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON market_orders (simulation, market_id, start_time);
CREATE INDEX ON market_orders (simulation, unit_id);
CREATE INDEX idx_market_orders_sim_unit_start
  ON market_orders (simulation, unit_id, start_time);

----------------------------
-- 4) unit_dispatch (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS unit_dispatch (
  simulation               TEXT    NOT NULL,
  time                     TIMESTAMP NOT NULL,
  unit                     TEXT,
  power                    REAL,
  heat                     REAL,
  soc                      REAL,
  energy_generation_costs  REAL,
  energy_cashflow          REAL,
  total_costs              REAL,
  PRIMARY KEY (simulation, time, unit)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON unit_dispatch (simulation, time);
CREATE INDEX idx_unit_dispatch_sim_unit_time
  ON unit_dispatch (simulation, unit, time);

----------------------------
-- 5.1) power_plant_meta (static)
----------------------------
CREATE TABLE IF NOT EXISTS power_plant_meta (
  simulation      TEXT    NOT NULL,
  unit_id         TEXT,
  unit_operator   TEXT,
  max_power       REAL,
  min_power       REAL,
  emission_factor REAL,
  efficiency      REAL,
  technology      TEXT,
  node            TEXT,
  PRIMARY KEY (simulation, unit_id)
);

CREATE INDEX idx_power_plant_meta_sim_tech
  ON power_plant_meta (simulation, technology);

----------------------------
-- 5.2) storage_meta (static)
----------------------------
CREATE TABLE IF NOT EXISTS storage_meta (
  simulation            TEXT    NOT NULL,
  unit_id               TEXT,
  unit_operator         TEXT,
  max_soc               REAL,
  min_soc               REAL,
  max_power_charge      REAL,
  max_power_discharge   REAL,
  min_power_charge      REAL,
  min_power_discharge   REAL,
  efficiency_charge     REAL,
  efficiency_discharge  REAL,
  technology            TEXT,
  node                  TEXT,
  PRIMARY KEY (simulation, unit_id)
);

----------------------------
-- 5.3) demand_meta (static)
----------------------------
CREATE TABLE IF NOT EXISTS demand_meta (
  simulation      TEXT    NOT NULL,
  unit_id         TEXT,
  unit_type       TEXT,
  unit_operator   TEXT,
  max_power       REAL,
  min_power       REAL,
  technology      TEXT,
  node            TEXT,
  PRIMARY KEY (simulation, unit_id)
);

----------------------------
-- 5.4) exchange_meta (static)
----------------------------
CREATE TABLE IF NOT EXISTS exchange_meta (
  simulation      TEXT    NOT NULL,
  unit_id         TEXT,
  unit_operator   TEXT,
  price_import    REAL,
  price_export    REAL,
  technology      TEXT,
  node            TEXT,
  PRIMARY KEY (simulation, unit_id)
);

----------------------------
-- 6) rl_params (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS rl_params (
  simulation         TEXT      NOT NULL,
  unit               TEXT,
  datetime           TIMESTAMP NOT NULL,
  evaluation_mode    BOOLEAN,
  episode            INTEGER,
  profit             REAL,
  reward             REAL,
  regret             REAL,
  critic_loss        REAL,
  total_grad_norm    REAL,
  max_grad_norm      REAL,
  learning_rate      REAL,
  PRIMARY KEY (simulation, episode, evaluation_mode, unit, datetime)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON rl_params (simulation, datetime);

----------------------------
-- 7) rl_meta (static)
----------------------------
CREATE TABLE IF NOT EXISTS rl_meta (
  simulation       TEXT    NOT NULL,
  episode          INTEGER,
  eval_episode     INTEGER,
  learning_mode    BOOLEAN,
  evaluation_mode  BOOLEAN,
  PRIMARY KEY (simulation, episode, evaluation_mode)
);

----------------------------
-- 8) grid_flows (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS grid_flows (
  simulation  TEXT    NOT NULL,
  datetime    TIMESTAMP NOT NULL,
  line        TEXT,
  flow        REAL,
  PRIMARY KEY (simulation, datetime)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON grid_flows (simulation, datetime);

----------------------------
-- 9) kpis (partitioned by simulation)
----------------------------
CREATE TABLE IF NOT EXISTS kpis (
  simulation  TEXT    NOT NULL,
  variable    TEXT,
  ident       TEXT,
  value       REAL,
  time        TIMESTAMP DEFAULT now(),
  PRIMARY KEY (simulation, variable, ident)
)
PARTITION BY LIST (simulation);

CREATE INDEX ON kpis (simulation, time);
