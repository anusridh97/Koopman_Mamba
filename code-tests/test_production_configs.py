from koopman_lm.config import build_config


def test_50m_production_config():
    c = build_config('50m')
    assert c.d_model == 384
    assert c.n_layers == 17
    assert c.ska_rank == 24
    assert tuple(c.ska_layer_indices) == (3, 7, 11, 15)
    assert c.ska_mode == 'parallel'
    assert c.ska_prefix_scan is True
    assert c.mlp_type == 'swiglu'


def test_180m_production_config():
    c = build_config('180m')
    assert c.d_model == 640
    assert c.n_layers == 25
    assert c.ska_rank == 24
    assert tuple(c.ska_layer_indices) == (3, 7, 11, 15, 19, 23)
    assert c.ska_mode == 'parallel'
    assert c.ska_prefix_scan is True
    assert c.mlp_type == 'swiglu'
