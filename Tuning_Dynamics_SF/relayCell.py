class relayCell:
    mouse: str = None
    id: str = None
    dsi: bool = False
    osi: float = None
    dir_fr: float = None
    sf_X: float = None
    sf_Y: float = None
    sf: float = None
    sf_cutoff: float = None
    tf_X: float = None
    tf_Y: float = None
    tf: float = None
    tf_bandwidth: float = None
    g1_xloc: float = None  # x1
    g1_yloc: float = None  # x2
    g2_xloc: float = None  # x3
    g2_yloc: float = None  # x4
    g1_xvar: float = None  # x5
    g1_rot: float = None  # x6 (clockwise)
    g1_yvar: float = None  # x7
    g2_xvar: float = None  # x8
    g2_rot: float = None  # x9 (clockwise)
    g2_yvar: float = None  # x10
    g1_tcen: float = None  # x11 ("ie centre at x11 ms before")
    g2_tcen: float = None  # x12 ("ie centre at x11 ms before")
    g1_tvar: float = None  # x13
    g2_tvar: float = None  # x14
    g1_tamp: float = None  # x15
    g2_tamp: float = None  # x16
    predicted_tfmax: float = None  # save pyLGN predicted peak response temporal frequency
