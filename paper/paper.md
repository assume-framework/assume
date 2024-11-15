<!--
SPDX-FileCopyrightText: ASSUME Developers

SPDX-License-Identifier: AGPL-3.0-or-later
-->

---
title: 'ASSUME - Agent-based Simulation for Studying and Understanding Market Evolution'
tags:
  - Python
  - agent based simulation
  - energy market
  - reinforcement learning
  - market simulation
  - simulation
authors:
  - name: Florian Maurer
    orcid: 0000-0001-8345-3889
    corresponding: true
    affiliation: 1
    "affiliation": 1
  - name: Nick Harder
    orcid: 0000-0003-1897-3671
    "affiliation": 2
  - name: Kim Kira Miskiw
    orcid: 0009-0009-1389-4844
    affiliation: 3
  - name: Manish Khanra
    orcid: 0000-0002-3347-9922
    affiliation: 4

affiliations:
 - name: University of Applied Sciences Aachen, Germany
   index: 1
 - name: Inatech Universität Freiburg, Germany
   index: 2
 - name: Karlsruhe Institute of Technology, Germany
   index: 3
 - name: Fraunhofer Institute for Software and Systems Engineering, Germany
   index: 4
date: 13 November 2024
bibliography: paper.bib
---

<!-- pandoc -s paper.md -o paper.pdf --bibliography paper.bib --csl=apa.csl --filter pandoc-citeproc --pdf-engine=xelatex-->

# Summary

**ASSUME** is an open-source toolbox for agent-based simulations of European electricity markets, with a primary focus on the German market setup. Developed as an open-source model, its primary objectives are to ensure usability and customizability for a wide range of users and use cases in the energy system modeling community.

A unique feature of the ASSUME toolbox is its integration of **Deep Reinforcement Learning** methods into the behavioral strategies of market agents. The model offers various predefined agent representations for both the demand and generation sides, which can be used as plug-and-play modules, simplifying the reinforcement of learning strategies. This setup enables research into new market designs and dynamics in energy markets.

# Statement of need

While different other agent-based models have been developed for the study of energy markets, such as PowerACE [@bublitzAgentbasedSimulationGerman2014] and AMIRIS [@schimeczekAMIRISAgentbasedMarket2023], the possible integration of reinforcement learning methods into the behavioral strategies of market agents is currently unique to ASSUME.
This feature allows for the exploration of new market designs and dynamics in energy markets using a common open-source simulation framework.
Further unique features of `ASSUME` are the extensive market abstraction which allows to define complex multi market scenarios as shown in [@maurerMarketAbstractionEnergy2023].
ASSUME is designed to provide results which are easily comparable and interactable.
This is set in contrast to other agent-based simulation frameworks, which are often designed to be used for a specific use case.
Instead, various moving parts can be configured, such as the market abstraction, the agent properties and bidding strategies, or the reinforcement learning methods.

# Architecture

`ASSUME` builds on the open-source agent-based simulation library [mango-agents](https://pypi.org/project/mango-agents/) to model the simulation world and to give users continuous feedback during the simulation runs.
Interaction with the results of the simulation is possible through the Grafana dashboards, which do have direct access to the simulation database.
Additionally, writing the output to CSV files is supported for scenarios in which a database is not available.

New scenarios can be created by providing CSV files, accessing simulation data from a database or by using the object-oriented API, which makes it possible to integrate new classes, maket mechanisms and bidding strategies as well.
The core overview of available classes and the interaction with the world is shown in \autoref{fig:architecture}.

![Basic Architecture of the ASSUME simulation\label{fig:architecture}](../docs/source/img/architecture.svg)

# Publications

ASSUME has been used to investigate the usage of complex order types like block bids in wholesale markets [@adamsBlockOrdersMatter2024] as well as for the integration of demand-side flexibility and redispatch markets in [@khanraEconomicEvaluationElectricity2024].
Studies of applicability of reinforcement learning methods in energy markets were tackled in [@harderHowSatisfactoryCan2024], while an analysis of explainable AI methods was appliedd to ASSUME in [@miskiwExplainableDeepReinforcement2024].
For better interoperability with other energy system model data, adapters to interact from ASSUME with PyPSA and AMIRIS are available and make a comparison to other renomated market simulation tools possible [@maurerKnowYourTools2024].

# Acknowledgements

Kim K. Miskiw, Nick Harder and Manish Khanra thank the German Federal Ministry for Economic Affairs and Climate Action for the funding of the ASSUME project under grant number BMWK 03EI1052A.

# References