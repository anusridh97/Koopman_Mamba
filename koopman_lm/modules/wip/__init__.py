"""Work-in-progress modules -- feature-complete but NOT yet wired into any
model or evaluated.

memory.py (LastLayerRidgeMemory / BOM-LM-v0): inference-only last-layer ridge
memory. The streaming wiring exists in modules/recurrent.py, but nothing
constructs it in a live eval path yet. Parked here pending a keep/strip
decision -- do not treat as production.
"""
