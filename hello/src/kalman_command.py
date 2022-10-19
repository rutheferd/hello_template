from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
import numpy as np
from datetime import datetime, timedelta
start_time = datetime.now()

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
from scipy.stats import uniform
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.plotter import Plotterly

def kalman(seed=123):

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

    truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=start_time)])
    for k in range(1, 21):
        truth.append(GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)))
    truths.add(truth)

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

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
    data_associator = GlobalNearestNeighbour(hypothesiser)
    prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
    prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

    tracks = {Track([prior1]), Track([prior2])}

    for n, measurements in enumerate(all_measurements):
        # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
        hypotheses = data_associator.associate(tracks,
                                            measurements,
                                            start_time + timedelta(seconds=n))
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
            else:  # When data associator says no detections are good enough, we'll keep the prediction
                track.append(hypothesis.prediction)

    plotter = Plotterly()
    plotter.plot_ground_truths(truths, [0, 2])
    plotter.plot_measurements(all_measurements, [0, 2])
    plotter.plot_tracks(tracks, [0, 2], uncertainty=True)
    plotter.fig.write_image('Tracks.png')

def run():
    kalman()