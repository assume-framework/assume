---
title: 'ASSUME - Agent-based Simulation for Studying and Understanding Market Evolution'
tags:
  - Python
  - agent based modeling
  - energy market
  - reinforcement learning
  - software simulation
authors:
  - name: Florian Maurer
    orcid: 0000-0001-8345-3889
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Nick Harder
    orcid: 0000-0003-1897-3671
    equal-contrib: true
    affiliation: 2
  - name: Kim K. Miskiw
    orcid: 0009-0009-1389-4844
    equal-contrib: true
    affiliation: 3
  - name: Manish Khanra
    orcid: 0000-0002-3347-9922
    equal-contrib: true
    affiliation: 4

affiliations:
 - name: University of Applied Sciences Aachen, Germany
   index: 1
 - name: Institute for Sustainable Systems Engineering, University of Freiburg, Freiburg, Germany
   index: 2
 - name: Institute of Information Systems and Marketing, Karlsruhe Institute of Technology, Karlsruhe, Germany
   index: 3
 - name: Fraunhofer Institute for Systems and Innovation Research, Karlsruhe, Germany
   index: 4
date: 13 November 2024
bibliography: paper.bib
---

# Summary

**ASSUME** is an open-source toolbox for agent-based simulations of various energy markets, with a primary focus on European electricity markets. Developed as an open-source model, its objectives are to ensure usability and customizability for a wide range of users and use cases in the energy system modeling community.

A unique feature of the `ASSUME` toolbox is its integration of **Deep Reinforcement Learning** methods into the behavioral strategies of market agents. The model offers various predefined agent representations for both the demand and generation sides, which can be used as plug-and-play modules, simplifying the usage of reinforcement learning strategies. This setup enables research of new market designs and evolving dynamics in energy markets.

# Statement of need

Various open-source agent-based models have been developed for studying energy markets, such as PowerACE [@bublitzAgentbasedSimulationGerman2014] and AMIRIS [@schimeczekAMIRISAgentbasedMarket2023].
Yet, the possible integration of reinforcement learning methods into the behavioral strategies of market agents is currently unique to `ASSUME` and is build upon prior research on multi-agent reinforcement learning [@harderFitPurposeModeling2023].
Simulations which solely rely on rule-based bidding strategy representation, limit the ability to represent future markets or alternative markets designs, as in reality bidding agents would adapt to the new market design.
Most notably, `ASSUME` enables the highest number of simultaneously learning market agents in literature [@miskiwExplainableDeepReinforcement2024].
This feature allows for the exploration of new market designs and emergent dynamics in energy markets using a common open-source simulation framework.
Further unique features of `ASSUME` are the extensive market abstraction which allows to define complex multi-market scenarios as shown by @maurerMarketAbstractionEnergy2023.
Even redispatch markets and nodal markets are supported, making it possible to represent network constraints and market coupling.
`ASSUME` is designed to provide results which are easily comparable and interactable.
This is set in contrast to other agent-based simulation frameworks, which are often designed to be used for a specific use case.
Instead, various moving parts can be configured, such as the market abstraction, the agent properties and bidding strategies, or the reinforcement learning methods.

# Architecture

`ASSUME` builds on the open-source agent-based simulation library [mango-agents](https://pypi.org/project/mango-agents/) to model the simulation world and to give users continuous feedback during the simulation runs.
Interaction with the results of the simulation is possible through the preconfigured provided Grafana dashboards, which do have direct access to the simulation database.
Additionally, writing the output to CSV files is supported for scenarios in which a database is not available or needed.

![Componental overview of the ASSUME simulation architecture\label{fig:architecture}](../docs/source/img/architecture.svg)

New scenarios can be created by providing CSV files, accessing simulation data from a database or by using the object-oriented API.
The object-oriented API makes it possible to integrate new classes, market mechanisms and bidding strategies as well.

Less technical users can adjust the examples from the yaml config and csv inputs directly, while new bidding strategies and methods can be implemented using the provided base classes.
This is possible through the decoupled architecture which separates the declaration of simulation behavior from the input data, while allowing to define either externally without changes to the core.
The overview of available classes and the interaction with the world is shown in \autoref{fig:architecture}.
Extensive documentation about these features is available at [https://assume.readthedocs.io/en/latest/](https://assume.readthedocs.io/en/latest/examples_basic.html).
This also includes a variety of notebooks which clarify the usage through examples.

# Publications

`ASSUME` has been used to investigate the usage of complex order types like block bids in wholesale markets [@adamsBlockOrdersMatter2024] as well as for the integration of demand-side flexibility and redispatch markets by @khanraEconomicEvaluationElectricity2024.
Studies of applicability of reinforcement learning methods in energy markets were tackled by @harderHowSatisfactoryCan2024, while an analysis of explainable AI methods was applied to `ASSUME` by @miskiwExplainableDeepReinforcement2024.
For better interoperability with other energy system model data, adapters to interact from `ASSUME` with PyPSA and AMIRIS are available and make a comparison to other renomated market simulation tools possible [@maurerKnowYourTools2024].

# Acknowledgements

This work was conducted as part of the project "ASSUME: Agent-Based Electricity Markets Simulation Toolbox," funded by the German Federal Ministry for Economic Affairs and Energy under grant number BMWK 03EI1052A.
We express our gratitude to all contributors to ASSUME.

# References
