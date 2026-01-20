# Energy Flow Analysis

## Components:
- **Grid**: External power source
- **PV**: Solar generation
- **Battery**: Energy storage (optional)
- **CS (Charging Station)**: Interface between grid and EV
- **EV**: Electric vehicle battery

## Energy Flows:

### CS.discharge (Şarj istasyonundan EV'ye):
```
Grid/PV/Battery → CS → CS.discharge → EV (şarj)
```
This is a LOAD (consumption)

### CS.charge (EV'den şarj istasyonuna - V2G):
```
EV (deşarj) → CS.charge → CS → Grid/Battery
```
This is a SUPPLY (generation from EV perspective)

## Electricity Balance Should Be:

### Supply Side (Sources):
1. grid_power (from external grid)
2. pv_used (from PV panels)
3. battery.discharge (from battery storage)
4. **CS.charge (from EV batteries via V2G)**

### Load Side (Sinks):
1. CS.discharge (charging EVs)
2. battery.charge (charging battery)
3. grid_feedin (exporting to grid)

## Equation:
```
grid_power + pv_used + battery.discharge + CS.charge
=
CS.discharge + battery.charge + grid_feedin
```

## Current Implementation (WRONG?):
```python
supply = grid_power + pv_used + battery.discharge
loads = CS.discharge + battery.charge + grid_feedin
```

**CS.charge is MISSING from supply side!**

This causes infeasibility when:
- EV wants to discharge (CS.charge > 0)
- But there's nowhere for that energy to go in the balance equation
