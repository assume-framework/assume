#!/usr/bin/env python3
"""
Comprehensive debug script to analyze queue logic consistency
"""

import pandas as pd
import pyomo.environ as pyo

def analyze_queue_constraints():
    """Analyze all queue-related constraints for logical consistency"""
    print("=== QUEUE LOGIC ANALYSIS ===\n")
    
    print("1. QUEUE CONSTRAINT ANALYSIS:")
    print("-" * 40)
    
    print("Current Queue Logic (enforce_queue_logic):")
    print("  Formula: in_queue[ev, t] = availability[t] - sum(assigned[ev, cs, t])")
    print("  Logic: EV is in queue if available but not assigned to any charging station")
    print("  Expected behavior:")
    print("    - Available=1, Assigned=0 -> Queue=1 (OK)")
    print("    - Available=1, Assigned=1 -> Queue=0 (OK)") 
    print("    - Available=0, Assigned=0 -> Queue=0 (OK)")
    print("    - Available=0, Assigned=1 -> Queue=-1 (IMPOSSIBLE!)")
    print()
    
    print("2. ASSIGNMENT CONSTRAINT INTERACTIONS:")
    print("-" * 40)
    
    constraints = [
        ("max_one_ev_per_station", "sum(assigned[ev, cs, t]) <= 1 for each cs"),
        ("one_assignment_per_ev", "sum(assigned[ev, cs, t]) <= 1 for each ev"),
        ("assignment_only_if_available", "assigned[ev, cs, t] <= availability[t]"),
        ("ensure_station_utilization", "total_assignments >= 1 if available_evs > 0"),
        ("strict_ev_priority", "second_ev assignment <= sufficient_charge[first_ev] + first_ev_elsewhere"),
        ("enhanced_first_ev_priority", "second_ev_total <= sufficient_charge[first_ev]"),
        ("prevent_ev_switching", "assignments[t] >= assignments[t-1] - sufficient_charge[t-1]")
    ]
    
    for name, formula in constraints:
        print(f"  {name}: {formula}")
    print()
    
    print("3. POTENTIAL LOGICAL CONFLICTS:")
    print("-" * 40)
    
    conflicts = []
    
    # Conflict 1: Over-constraining assignment
    print("Conflict 1: Multiple Priority Constraints")
    print("  - strict_ev_priority (line 674-679)")
    print("  - enhanced_first_ev_priority (line 747-748)")
    print("  - ensure_station_utilization (line 708-709)")
    print("  Problem: May over-constrain and create infeasibility")
    print()
    
    # Conflict 2: Queue calculation assumptions
    print("Conflict 2: Queue Logic Assumptions")
    print("  Current: in_queue = available - assigned")
    print("  Assumes: An EV can only be in one state (assigned OR queue)")
    print("  Problem: What if assignment constraints force assignment=0 but EV is available?")
    print("  Then: in_queue = 1 - 0 = 1 (goes to queue) - might be unintended")
    print()
    
    # Conflict 3: Sufficient charge logic
    print("Conflict 3: Sufficient Charge Determination")
    print("  Current: Based on future_usage_total and other_evs_available_time")
    print("  Problem: Complex calculation might give wrong 'sufficient' flag")
    print("  Impact: If flag is wrong, priority constraints misbehave")
    print()
    
    print("4. SUGGESTED SCENARIOS TO TEST:")
    print("-" * 40)
    
    scenarios = [
        "Both EVs available, both need charging (timestep=0)",
        "EV1 charging, EV2 available and needs charging",
        "EV1 sufficient charge, EV2 available and needs charging", 
        "Both EVs available, charging station occupied by EV1",
        "EV1 not available (driving), EV2 available and needs charging",
        "Both EVs have sufficient charge, both available"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"  Scenario {i}: {scenario}")
    print()
    
    print("5. RECOMMENDED FIXES:")
    print("-" * 40)
    
    recommendations = [
        "Simplify priority constraints - remove redundant ones",
        "Add bounds checking for queue calculation",
        "Make sufficient_charge_for_travel calculation more robust",
        "Add constraint validation to prevent impossible states",
        "Test edge cases with manual constraint verification"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    print()

def analyze_constraint_priorities():
    """Analyze constraint priority and potential conflicts"""
    print("=== CONSTRAINT PRIORITY ANALYSIS ===\n")
    
    print("Current Constraint Hierarchy (order of enforcement):")
    print("1. Physical constraints (max_one_ev_per_station, one_assignment_per_ev)")
    print("2. Availability constraints (assignment_only_if_available)")
    print("3. Priority constraints (strict_ev_priority, enhanced_first_ev_priority)")
    print("4. Utilization constraints (ensure_station_utilization)")
    print("5. Queue calculation (enforce_queue_logic)")
    print()
    
    print("Potential Issues:")
    print("- Priority constraints might conflict with utilization constraints")
    print("- Queue logic assumes assignment constraints are always satisfiable")
    print("- No validation that sufficient_charge flags are correctly calculated")
    print()

def check_mathematical_consistency():
    """Check mathematical consistency of constraint system"""
    print("=== MATHEMATICAL CONSISTENCY CHECK ===\n")
    
    print("Queue Equation Validation:")
    print("Formula: in_queue[ev,t] = availability[ev,t] - sum(assigned[ev,cs,t])")
    print()
    
    print("Valid ranges:")
    print("- availability[ev,t] ∈ {0, 1}")
    print("- assigned[ev,cs,t] ∈ {0, 1}")
    print("- sum(assigned[ev,cs,t]) ∈ {0, 1} (due to one_assignment_per_ev)")
    print("- Therefore: in_queue[ev,t] ∈ {0, 1} ✓")
    print()
    
    print("Edge case check:")
    print("Case 1: available=1, sum(assigned)=0 -> queue=1 (OK)")
    print("Case 2: available=1, sum(assigned)=1 -> queue=0 (OK)")
    print("Case 3: available=0, sum(assigned)=0 -> queue=0 (OK)")
    print("Case 4: available=0, sum(assigned)=1 -> queue=-1 (ERROR)")
    print()
    print("ISSUE: Case 4 is impossible due to assignment_only_if_available constraint")
    print("But queue variable is not bounded, so solver might assign negative values")
    print()
    
    print("RECOMMENDATION: Add explicit bounds to queue variable")
    print("queue[ev,t] ∈ {0, 1} and add constraint: queue[ev,t] >= 0")

if __name__ == "__main__":
    analyze_queue_constraints()
    analyze_constraint_priorities()
    check_mathematical_consistency()
    
    print("\n" + "="*60)
    print("SUMMARY OF ISSUES FOUND:")
    print("="*60)
    print("1. Multiple redundant priority constraints may cause conflicts")
    print("2. Queue variable not explicitly bounded")
    print("3. Sufficient charge calculation complexity may cause errors")
    print("4. No validation of constraint satisfiability")
    print("5. Need to test specific scenarios to verify behavior")