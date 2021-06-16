config_pattern_mining = {
    'target_col':'components.cont.conditions.logic.errorCode',
    'min_gap_since_last_error':1000,
    'min_obs_since_last_error':10,
    'window_len':1000,
    'cross_component':True,
    'support':0.5,
    'confidence':0.5}