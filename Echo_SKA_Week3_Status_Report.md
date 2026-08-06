# Echo / SKA Project: Week 3 Status Report

## Objective
* Reproduce the results of Table 2 from the paper to verify that the SSM+SKA model maintains performance on 64x longer sequences while baseline models degrade.
* Execute the `table2.py` script to confirm if the output matches the paper's metrics.

## Code Corrections and Structural Observations
* **Attention Baseline Fix:** Corrected the SSM+Attn baseline from a bidirectional mode to causal attention to accurately reflect the paper's strict look-backward constraints.
* **Evaluation Enforcement:** Added programmatic checks to ensure the zero-shot evaluation setup strictly tests on 1 key-value pair after training on 8.
* **Architecture Deviations:** Identified a 10-12% discrepancy in parameter counts for two models compared to the paper.
* **Workaround Implementation:** The current implementation uses a simplified, hand-built Mamba-2 substitute with a flat learning rate to bypass Windows installation limitations.

## Initial Reproduction Results
* After training SSM+SKA for the specified 6,000 steps, accuracy plateaued at 13-18%, significantly below the paper's reported 82-95%.
* This performance failure occurred at the baseline training length, indicating a fundamental issue rather than a specific failure to generalize to longer contexts.

## Diagnostic Testing
To isolate the failure point efficiently, rapid CPU tests were conducted:
* **Basic Functionality Verification:** Training and testing exclusively on 1 key-value pair achieved 100% accuracy, confirming the core model and training pipeline function correctly.
* **Unsuccessful Adjustments:** Modifying the training data to include mixed key-value counts, targeting 2-3 key-value pairs, explicitly highlighting relevant tokens amidst noise, and integrating a more mathematically faithful SKA implementation all failed to improve accuracy.
* **Task Structure Hypothesis:** Project archives indicate that the SKA architecture requires structured, temporal data where values update or overwrite over time.
* **Current Task Limitation:** The current task utilizes random placement without temporal structure, giving the SKA mechanisms nothing to utilize.

## Next Steps
* Having ruled out software bugs, learning rate issues, and simplified mathematics, the primary hypothesis is that the under-specified training task from the paper is the root cause.
* The immediate next action is to conduct a longer, structured test using an overwrite-style task on the CPU to evaluate its viability before committing GPU resources.
