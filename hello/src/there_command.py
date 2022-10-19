import logging
import numpy as np
from datetime import datetime, timedelta
start_time = datetime.now()
import pickle

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def generate(seed, plot, path='./generate.png'):
    logging.info("Generating Data...")
    np.random.seed(seed)

    truths = set()

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                            ConstantVelocity(0.005)])

    truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
    for k in range(1, 21):
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)))      
    truths.add(truth)
    print(type(truth))

    truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=start_time)])
    for k in range(1, 21):
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)))
    truths.add(truth)
    logging.info(truths)
    with open('kos.txt','wb') as f:
        pickle.dump(truths, f)

    if plot:
        logging.info("Plotting...")
        from stonesoup.plotter import Plotterly
        plotter = Plotterly()
        plotter.plot_ground_truths(truths, [0, 2])
        plotter.fig.write_image(path)


def run(seed, plot):
    generate(seed, plot)
