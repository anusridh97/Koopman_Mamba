"""§5.4: the trainer installs a SIGUSR1 handler that flags for a clean exit
(train.py's loop then writes resume.pt and returns) rather than dying wherever
Slurm's warning signal happens to land mid-step.
"""
import os
import signal
import time

import pytest

pytestmark = pytest.mark.correctness


def test_sigusr1_sets_the_preemption_flag():
    from koopman_lm.training.train import PreemptionFlag, install_sigusr1_handler

    flag = PreemptionFlag()
    assert not flag.is_set()
    install_sigusr1_handler(flag)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        # signal delivery to the main thread happens between bytecode
        # instructions; give it a moment
        for _ in range(100):
            if flag.is_set():
                break
            time.sleep(0.01)
        assert flag.is_set()
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
