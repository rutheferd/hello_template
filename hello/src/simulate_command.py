from importlib.resources import path
from scipy.stats import uniform
import logging
from stonesoup.plotter import Plotterly

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
import numpy as np
import ast
import pickle

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

def simulate(truths_path, save_path='./sim.png'):

    logging.info("Reading File...")
    with open(truths_path,'rb') as f:
        truths = pickle.load(f)

    measurement_model = LinearGaussian(
        ndim_state=4,
        mapping=(0, 2),
        noise_covar=np.array([[0.75, 0],
                            [0, 0.75]])
        )
    all_measurements = []

    for k in range(20):
        measurement_set = set()

        for truth in truths:
            # Generate actual detection from the state with a 10% chance that no detection is received.
            if np.random.rand() <= 0.9:
                measurement = measurement_model.function(truth[k], noise=True)
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                groundtruth_path=truth,
                                                timestamp=truth[k].timestamp,
                                                measurement_model=measurement_model))

            # Generate clutter at this time-step
            truth_x = truth[k].state_vector[0]
            truth_y = truth[k].state_vector[2]
            for _ in range(np.random.randint(10)):
                x = uniform.rvs(truth_x - 10, 20)
                y = uniform.rvs(truth_y - 10, 20)
                measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                            measurement_model=measurement_model))
        all_measurements.append(measurement_set)
    
    plotter = Plotterly()
    plotter.plot_ground_truths(truths, [0, 2])
    plotter.plot_measurements(all_measurements, [0, 2])
    logging.info(all_measurements)
    plotter.fig.write_image(save_path)
    logging.info("Plot Image Saved!")

def run(truths_path, save_path):
    simulate(truths_path=truths_path, save_path=save_path)
    logging.info("Done!")