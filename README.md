# ASSUME

The ASSUME Framework is a software framework to simulate energy markets and the behavior of market agents.

It features options to use Reinforcement Learning Agents.

## Get started

First create a conda env (alternativly a venv)

```
conda create -n assume python=3.9
conda activate assume
```

then install the assume framework and its dependencies:

```
pip install -e .
```

and run the example market simulation:

```
python examples/example_01.py
```

## Get started using docker

You can also start the project using 

```
docker compose up --build
```

This will start a container for timescaledb, a preconfigured grafana dashboard and run the simulation `example_01.py` in a containerized environment

## Convention

| param | description |
|-------|-------------|
| t     | time        |
