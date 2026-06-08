"""Separable Field Cellular Automaton (SFCA).

A batched GPU simulator for the rank-1 separable-field CA studied in
"Boundary-Mediated Metastable Dynamics in a Separable Field Cellular
Automaton".

Public entry points:
    sfca.simulate.simulate_batch   -- batched simulator with outcome detection
    sfca.rule.update_batch         -- single update step (batched)
"""
from .simulate import simulate_batch  # noqa: F401
from .rule import update_batch        # noqa: F401
