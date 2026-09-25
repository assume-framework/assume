import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# According to https://cd.uni-freiburg.de/farben/
blau = "#344A9A"
dunkelblau = "#00004a"
sand = "#f6f1e3"
gruen = "#00a082"
braun = "#8f6b30"
gelb = "#ffe863"
rosa = "#f5c2ed"
schwarz = "#000000"

COLORS = dict()
COLORS["nuclear_plant"] = gruen
COLORS["gas_plant_marginal"] = rosa
COLORS["marginal"] = rosa
COLORS["gas_plant_learning"] = dunkelblau
COLORS["learning"] = dunkelblau
COLORS["back_up_plant"] = braun
COLORS["backup_plant"] = braun
COLORS["demand"] = schwarz

COLORS["gas_plant_learning_1"] = blau
COLORS["gas_plant_learning_2"] = dunkelblau


def plot_merit_order(df: pd.DataFrame):

  if not "marginal_costs" in df.columns:
    msg = "power_plants requires column 'marginal_costs'."
    raise ValueError(msg)

  sns.set_theme(style="whitegrid")

  # Aufsteigend nach Grenzkosten sortieren
  df = df.sort_values("marginal_costs").reset_index(drop=True)

  # X-Startpunkte für jeden Balken berechnen
  df["x_start"] = df["max_power"].cumsum() - df["max_power"]

  # Plot erstellen
  fig, ax = plt.subplots(figsize=(9, 5))

  palette = sns.color_palette("colorblind", n_colors=len(df))

  # Balken mit individueller Breite zeichnen
  for i, row in df.iterrows():
      ax.bar(
          x=row["x_start"],
          height=row["marginal_costs"],
          width=row["max_power"],
          align="edge",
          label=row["name"],
          color=COLORS[row["name"]],
          edgecolor="black",
          alpha=0.8,
      )


  ax.set_xlabel("Generation (MW)")
  ax.set_ylabel("Marginal Cost (€/MWh)")
  ax.set_title("Merit-Order")
  ax.legend(title="Power Plant")

  plt.tight_layout()

  return fig, ax

def plot_demand(df: pd.DataFrame):
  # Unpack both figure and axis objects
  fig, ax = plt.subplots(figsize=(10, 5))

  # Plot data
  sns.lineplot(data=df, x="datetime", y="demand_EOM", ax=ax, color=schwarz)

  # Format x-axis hours
  date_form = mdates.DateFormatter("%H")
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  ax.set(xlabel="Time", ylabel="Demand (MW)")
  ax.set_title("Inflexible Demand Curve (EOM)")

  index_below_1000 = df[df["demand_EOM"] < 1000]
  
  sns.scatterplot(
      data=index_below_1000,
      x="datetime",
      y=1000,
      ax=ax,
      s=25,
      marker="s",
      edgecolor=None,
      label="Demand <= 1000 MW",
      color=gelb) # drawstyle="steps-post")
  # ax.hlines(y=1000, xmin=index_below_1000.min())

  plt.tight_layout()

  plt.show()

  return fig, ax

def plot_accepted_volume(df: pd.DataFrame):
  date_form = mdates.DateFormatter("%H")

  df["end_time"] = pd.to_datetime(df["end_time"])
  df["accepted_volume (abs)"] = (df["accepted_volume"].abs())
  
  fig, ax = plt.subplots(1)
  df_demand = df.query('unit_id == "demand_EOM"')
  df_generation = df.query('unit_id != "demand_EOM"')

  x_values = df_demand["end_time"]
  previous = [0] * len(x_values)
  for label in df_generation["unit_id"].unique():
    data = df_generation.query('unit_id == @label')
    y_values = data["accepted_volume (abs)"]
    # sns.lineplot(data=data, x="end_time", y="accepted_volume (abs)", label=label, color=COLORS[label])
    ax.plot(x_values, previous, label=label, color=COLORS[label])
    ax.fill_between(x_values, previous, previous + y_values, color=COLORS[label], alpha=0.45)
    previous = [prev + y for prev, y in zip(previous, y_values)]

  ax.plot(x_values, df_demand["accepted_volume (abs)"], label="demand", color=COLORS["demand"])
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  
  
  ax.set_ylim(bottom=0)
  ax.set_xlabel("Time")
  ax.set_ylabel("Accepted Volume (MWh)")
  
  ax.legend(title="Power Plant")
  plt.tight_layout()

  return fig, ax

def plot_revenue(df: pd.DataFrame):

  fig, ax = plt.subplots(1)

  df["end_time"] = pd.to_datetime(df["end_time"])
  df["revenue"] = df['accepted_price'] * df['accepted_volume'] 
  df["revenue (cum)"] = df.groupby("strategy")["revenue"].cumsum()

  sns.lineplot(data=df, x="end_time", y="revenue (cum)", ax=ax, hue="strategy", palette=COLORS)

  ax.set_ylabel("Revenue (€)")
  ax.set_xlabel("Time (h)")
  ax.legend(title="Strategy")

  date_form = mdates.DateFormatter('%H')
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

  return fig, ax


def plot_biddings(df: pd.DataFrame):
  fig, ax = plt.subplots()
  df["end_time"] = pd.to_datetime(df["end_time"])
  sns.scatterplot(df, x="end_time", y="price", hue="unit_id", ax=ax, marker="X", palette=COLORS)
  date_form = mdates.DateFormatter('%H')
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  ax.set_ylabel("Offer price (€)")
  ax.set_xlabel("Time (h)")
  ax.legend(title="Strategy")
  plt.tight_layout()

  return fig, ax


def plot_biddings_with_clearing_price(df: pd.DataFrame):
  fig, ax = plt.subplots()
  df["end_time"] = pd.to_datetime(df["end_time"])
  sns.scatterplot(df, x="end_time", y="price", hue="unit_id", ax=ax, marker="X", palette=COLORS)
  date_form = mdates.DateFormatter('%H')
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  ax.set_ylabel("Offer price (€)")
  ax.set_xlabel("Time (h)")
  ax.legend(title="Strategy")

  clearing_price_df = df.groupby('end_time')['accepted_price'].first().reset_index()
  sns.lineplot(clearing_price_df, y="accepted_price", x="end_time", color=gelb, alpha=0.8, label="clearing price")

  plt.tight_layout()

  return fig, ax