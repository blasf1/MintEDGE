from .services import Service
from .users import User, Stationary, Car, Person
from .mobility import Location, MobilityManager, RandomMobilityManager
from .energy import (
    EnergyAware,
    EnergyMeasurement,
    EnergyMeter,
    EnergyModelServer,
    EnergyModelLink,
)
from .infrastructure import Infrastructure, EdgeServer, BaseStation, Link
from .demand_predictor import IdealPredictor
from .allocation_strategy import AllocationStrategy
from .orchestrator import Orchestrator
from .simulation import Simulation

SEED = 0
RAND_NUM_GEN = None  # type: ignore
SIMULATION_TIME = None  # type: ignore
