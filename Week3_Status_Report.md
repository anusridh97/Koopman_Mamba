# Echo / SKA Project: Week 3 Status Report

## Objective
* Reproduce the results of Table 2 from the paper to verify that the SSM+SKA model maintains performance on 64x longer sequences while baseline models degrade.
* Execute the `table2.py` script to confirm if the output matches the paper's metrics.

## Code Corrections and Structural Observations
* **Attention Baseline Fix:** Corrected the SSM+Attn baseline from a bidirectional mode to causal attention to accurately reflect the paper's strict look-backward constraints.
* **Evaluation Enforcement:** Added programmatic checks to ensure the zero-shot evaluation setup strictly tests on 1 key-value pair after training on 4.
* **Architecture Deviations:** Identified a 10-12% discrepancy in parameter counts for two models compared to the paper.
* **Model Stack Update:** `table2.py` now builds models through the project's production implementation (`koopman_lm.models.baselines`), using the `mamba_ssm` Mamba-2 backbone and a more complete SKA module already present in the legacy code base. This also restored the paper's actual training recipe (AdamW with beta2=0.95, cosine learning-rate schedule with warmup, FP16 mixed precision).
* **Test Suite Updated:** Rewrote the test file to match; tests that need an actual model are marked to auto-skip on a CPU-only machine and will run for real the first time this executes in Colab.

## Initial Reproduction Results
* Before implementing the fixes training SSM+SKA for the specified 6,000 steps, accuracy plateaued at 13-18%, significantly below the paper's reported 82-95%.
* This performance failure occurred at the baseline training length, indicating a fundamental issue rather than a specific failure to generalize to longer contexts.
* This run predates the model stack update above; a fresh run with the current implementation has not yet been executed — that's the immediate next step.

## Diagnostic Testing
To isolate the failure point efficiently, rapid CPU tests were conducted:
* **Basic Functionality Verification:** Training and testing exclusively on 1 key-value pair achieved 100% accuracy, confirming the core model and training pipeline function correctly.
* **Unsuccessful Adjustments:** Modifying the training data to include mixed key-value counts, targeting 2-3 key-value pairs, explicitly highlighting relevant tokens amidst noise, and integrating a more mathematically faithful SKA implementation all failed to improve accuracy.
* **Task Structure Hypothesis:** Project archives indicate that the SKA architecture requires structured, temporal data where values update or overwrite over time.
* **Current Task Limitation:** The current task utilizes random placement without temporal structure, giving the SKA mechanisms nothing to utilize.
* These findings concern the training task rather than model implementation details, so they likely still apply going forward -- but that hasn't been confirmed yet.

## Next Steps
* Run the updated test suite in Colab to confirm the current implementation builds and trains end-to-end (not yet verified on this machine).
* Re-run the full 6,000-step SSM+SKA training with the current implementation and compare against the paper's 82-95% target -- this may close some or all of the gap on its own, independent of the task-structure question below.
* If the gap persists, follow up on the task-structure hypothesis: conduct a longer, structured test using an overwrite-style task before committing further GPU resources.
