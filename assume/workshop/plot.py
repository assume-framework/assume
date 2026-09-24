import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
          color=palette[i],
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
  sns.lineplot(data=df, x="datetime", y="demand_EOM", ax=ax)

  # Format x-axis hours
  date_form = mdates.DateFormatter("%H")
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  ax.set(xlabel="Time", ylabel="Demand (MW)")
  ax.set_title("Inflexible Demand Curve (EOM)")

  index_below_1000 = df[df["demand_EOM"] < 1000]
  print(index_below_1000.head())
  sns.scatterplot(
      data=index_below_1000,
      x="datetime",
      y=1000,
      ax=ax,
      s=25,
      marker="s",
      edgecolor=None,
      label="Demand <= 1000 MW",
      color="orange") # drawstyle="steps-post")
  # ax.hlines(y=1000, xmin=index_below_1000.min())

  plt.tight_layout()

  plt.show()

  return fig, ax

def plot_accepted_volume(df: pd.DataFrame):
  date_form = mdates.DateFormatter("%H")
  
  fig, ax = plt.subplots(1)
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
  df["end_time"] = pd.to_datetime(df["end_time"])
  
  sns.lineplot(data=df, x="end_time", y="accepted_volume", hue="unit_id", alpha=0.75)

  ax.set_xlabel("Time")
  ax.set_ylabel("Accepted Volume (MWh)")
  ax.set_ylim(-10, 1100)
  ax.legend(title="Power Plant")
  plt.tight_layout()

  return fig, ax

def plot_revenue(df: pd.DataFrame):

  fig, ax = plt.subplots(1)

  df["end_time"] = pd.to_datetime(df["end_time"])
  df["revenue"] = df['accepted_price'] * df['accepted_volume'] 
  df["revenue (cum)"] = df.groupby("strategy")["revenue"].cumsum()

  sns.lineplot(data=df, x="end_time", y="revenue (cum)", ax=ax, hue="strategy")

  ax.set_ylabel("Revenue (€)")
  ax.set_xlabel("Time (h)")
  ax.legend(title="Strategy")

  date_form = mdates.DateFormatter('%H')
  ax.xaxis.set_major_formatter(date_form)
  ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))

  return fig, ax