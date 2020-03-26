import logging
import time

import numpy as np

from eda import ma_data
from sir_fitting_us import seir_experiment, make_csv_from_traj

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Fitting model.")

# initial values taken from previous fit, used to seed MH sampler efficiently.
x0 = np.array([-0.19, -2.65, -3.21, -6.12, -19.60])
traj = seir_experiment(ma_data, x0, iterations=100)

mean_ll = np.mean([ll for (x, ll) in traj])
logger.info("Model fitting finished with mean log-likelihood: {}".format(mean_ll))

if mean_ll < -20:
    raise AssertionError(
        """Mean log-likelihood {} less than threshold of
        -20.  This is probably an error.""".format(mean_ll)
    )

underscored_time = time.ctime().replace(" ", "_")
fname = "ma_seir_output_{}.csv".format(underscored_time)
make_csv_from_traj(traj, ma_data, fname)
