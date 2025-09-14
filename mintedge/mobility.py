import math
import multiprocessing

import libsumo

from simpy.core import Environment
from tqdm import tqdm

import settings
from mintedge import Car, Person, Stationary


class Location:
    __slots__ = ["x", "y"]

    def __init__(self, x: float, y: float) -> None:
        """This class represents a geographical location.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.x = x
        self.y = y

    def distance(self, loc: "Location") -> float:
        """Distance between two locations using the Haversine formula.

        Args:
            loc (Location): Location to which the distance is calculated.
        Returns:
            float: Distance between the two locations.
        """
        R = 6371  # radius of Earth in kilometers
        phi1 = math.radians(loc.y)
        phi2 = math.radians(self.y)
        delta_phi = math.radians(self.y - loc.y)
        delta_lambda = math.radians(self.x - loc.x)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c * 1000  # convert to meters

    def __repr__(self):
        return f"Location({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class MobilityManager:
    __slots__ = ["env", "running_users", "users_sliding_window"]

    def __init__(self, env: Environment):
        """Class that manages the mobility of users.

        Args:
            env (Environment): The simulation environment.
        """
        self.env = env
        self._launch_sumo()
        self.running_users = {}
        # Initialize sliding window
        self.users_sliding_window = []
        self._initialize_sliding_window()

    def __del__(self):
        # Mobility finished
        libsumo.close()

    def _launch_sumo(self):
        """
        Launches the SUMO simulator with the given configuration.
        """
        # Leverage sumo for mobility
        from mintedge import SIMULATION_TIME as sim_time

        libsumo.start(
            [
                "sumo",
                "--net-file",
                settings.NET_FILE,
                "--route-files",
                settings.ROUTES_FILE,
                "--no-internal-links",
                "--ignore-route-errors",
                "--begin",
                "0",
                "--end",
                str(sim_time),
                "--no-warnings",
                "--route-steps",
                str(settings.ORCHESTRATOR_INTERVAL),
                "--threads",
                str(multiprocessing.cpu_count()),
                "--device.rerouting.threads",
                str(multiprocessing.cpu_count()),
                "--step-method.ballistic",
                "--random",
            ]
        )

    def run(self, env: Environment, infr: "Infrastructure"):
        """Starts the mobility manager process.

        Args:
            env (Environment): Simulation environment.
        """
        while True:
            users = self._get_next_step()
            for user, loc in users.items():
                if user in self.running_users:
                    self.running_users[user] = loc
                elif user.startswith("car"):
                    env.process(Car(env, user, infr, self, loc, env.now).run(env))
                    self.running_users[user] = loc
                elif user.startswith("person"):
                    env.process(Person(env, user, infr, self, loc, env.now).run(env))
                    self.running_users[user] = loc
                elif user.startswith("stationary"):
                    env.process(
                        Stationary(env, user, infr, self, loc, env.now).run(env)
                    )
                    self.running_users[user] = loc
            #
            dead = set(self.running_users.keys()) - set(users.keys())
            for d in dead:
                del self.running_users[d]
            yield env.timeout(1)

    # SUMO traces do not have stationary users
    def _get_next_step(self):
        """Returns the next step of the sliding window.
        Returns:
            dict: Dictionary with the user IDs as keys and their locations as values.
        """
        window_slot = {}
        for car in libsumo.vehicle.getIDList():
            lon, lat = libsumo.vehicle.getPosition(car)
            # UTM to WGS84
            lon, lat = libsumo.simulation.convertGeo(lon, lat)
            window_slot[f"car_{car}"] = Location(lon, lat)

        for person in libsumo.person.getIDList():
            lon, lat = libsumo.person.getPosition(person)
            # UTM to WGS84
            lon, lat = libsumo.simulation.convertGeo(lon, lat)
            window_slot[f"person_{person}"] = Location(lon, lat)

        self.users_sliding_window.append(window_slot)
        libsumo.simulationStep()
        try:
            users = self.users_sliding_window.pop(0)
        except IndexError:
            users = {}

        return users

    def _initialize_sliding_window(self):
        """
        Initializes the sliding window with the first positions of all users.
        """
        for _ in range(settings.ORCHESTRATOR_INTERVAL):
            window_slot = {}
            for car in libsumo.vehicle.getIDList():
                lon, lat = libsumo.vehicle.getPosition(car)
                # UTM to WGS84
                lon, lat = libsumo.simulation.convertGeo(lon, lat)
                window_slot[f"car_{car}"] = Location(lon, lat)

            for person in libsumo.person.getIDList():
                lon, lat = libsumo.person.getPosition(person)
                # UTM to WGS84
                lon, lat = libsumo.simulation.convertGeo(lon, lat)
                window_slot[f"person_{person}"] = Location(lon, lat)

            self.users_sliding_window.append(window_slot)
            libsumo.simulationStep()

    def get_user_location(self, user_id: str) -> Location:
        """Returns the location of a user.

        Args:
            user_id (str): The ID of the user whose location is returned.

        Return:
            Location: The location of the user.
        """

        return self.running_users[user_id]

    def get_running_user_count(self) -> int:
        """Returns the user count.

        Returns:
            int: User count.
        """
        return len(self.running_users)


class RandomMobilityManager(MobilityManager):
    def _launch_sumo(self):
        # Leverage sumo for mobility
        from mintedge import SIMULATION_TIME as sim_time

        libsumo.start(
            [
                "sumo",
                "--net-file",
                settings.NET_FILE,
                "--no-internal-links",
                "--ignore-route-errors",
                "--begin",
                "0",
                "--end",
                str(sim_time),
                "--no-warnings",
                "--route-steps",
                str(settings.ORCHESTRATOR_INTERVAL),
                "--threads",
                str(multiprocessing.cpu_count()),
                "--device.rerouting.threads",
                str(multiprocessing.cpu_count()),
                "--step-method.ballistic",
                "--random",
            ]
        )

    def _get_user_count(self, time: float, max_users: int):
        """Average number of users at this time step. This is passed to a
        Poisson distribution to get the actual number of users.

        Args:
            time (float): Time in seconds since the start of the simulation.
            max_users (int): Maximum number of users.

        Returns:
            float: Average number of users.
        """
        if (
            settings.USER_COUNT_DISTRIBUTION is None
            or settings.USER_COUNT_DISTRIBUTION == []
        ):
            return max_users

        hour = time // 3600 % len(settings.USER_COUNT_DISTRIBUTION)
        minim = time // 60 % 60
        user_dist = settings.USER_COUNT_DISTRIBUTION[int(hour)]
        user_dist_next = settings.USER_COUNT_DISTRIBUTION[
            int((hour + 1) % len(settings.USER_COUNT_DISTRIBUTION))
        ]  # make it cyclic (makes sense for daily/hourly/weekly patterns)
        # users (dis)appear little by little not all of a sudden
        decay_factor = (user_dist - user_dist_next) / 60
        avg_user_count = (user_dist - decay_factor * minim) * max_users

        return int(avg_user_count)

    def _get_next_step(self):
        """ "
        Returns the next step of the sliding window. It also creates or removes
        users if necessary.

        Returns:
            dict: Dictionary with the user IDs as keys and their locations as values.
        """
        # Check the number of cars and create more if necessary
        cars = self._get_user_count(self.env.now, settings.NUMBER_OF_CARS)
        if cars > libsumo.vehicle.getIDCount():
            cars_to_add = cars - libsumo.vehicle.getIDCount()
            for _ in range(cars_to_add):
                self._create_random_car(
                    int(self.env.now) + len(self.users_sliding_window)
                )
        # Check the number of people and create more if necessary
        people = self._get_user_count(self.env.now, settings.NUMBER_OF_PEOPLE)
        if people > libsumo.person.getIDCount():
            people_to_add = people - libsumo.person.getIDCount()
            for _ in range(people_to_add):
                self._create_random_person(
                    int(self.env.now) + len(self.users_sliding_window)
                )

        window_slot = {}
        window_slot = self._get_slot_cars(window_slot)
        window_slot = self._get_slot_people(window_slot)
        window_slot = self._get_slot_stationary(window_slot)

        self.users_sliding_window.append(window_slot)
        libsumo.simulationStep()
        try:
            users = self.users_sliding_window.pop(0)
        except IndexError:
            users = {}

        return users

    def _get_random_edge(self) -> str:
        """Returns a random edge in the map.
        Returns:
            str: Random edge ID.
        """
        from mintedge import RAND_NUM_GEN as random

        return random.choice(libsumo.edge.getIDList())  # type: ignore

    def _create_random_car(self, depart: int):
        """Creates a random car that departs at the given time.

        Args:
            depart (int): Departure time
        """
        time = libsumo.simulation.getTime()
        count = len(libsumo.simulation.getLoadedIDList())
        iden = f"{int(time)}_{count}"
        _, stage, _ = self._get_random_stage(depart)
        libsumo.route.add(routeID=f"route_{iden}", edges=stage.edges)
        libsumo.vehicle.add(vehID=f"car_{iden}", routeID=f"route_{iden}")

    def _create_random_person(self, depart: int):
        """Creates a random person that departs at the given time.

        Args:
            depart (int): Departure time
        """
        time = libsumo.simulation.getTime()
        count = len(libsumo.simulation.getLoadedIDList())
        iden = f"{int(time)}_{count}"
        src_edge, stage, _ = self._get_random_stage(depart)

        libsumo.person.add(
            personID=f"person_{iden}", edgeID=src_edge, pos=0, depart=depart
        )
        try:
            libsumo.person.appendWalkingStage(
                personID=f"person_{iden}", edges=stage.edges, arrivalPos=-1
            )
        except libsumo.libsumo.TraCIException:
            libsumo.person.remove(f"person_{iden}")
            return

    def _get_random_position(self) -> Location:
        """Returns a random position in the map.

        Returns:
            Location: Random position.
        """
        from mintedge import RAND_NUM_GEN as random

        edge = self._get_random_edge()
        shape = libsumo.lane.getShape(f"{edge}_0")
        lon, lat = random.choice(shape)  # type: ignore
        # UTM to WGS84
        lon, lat = libsumo.simulation.convertGeo(lon, lat)
        return Location(lon, lat)

    def _get_random_stage(self, depart: int):
        """Returns a random stage in the map.
        Args:
            depart (int): Departure time
        Returns:
            str, libsumo.TraCIStage, str: Source edge ID, stage, destination edge ID.
        """
        while True:
            dst_edge = self._get_random_edge()
            src_edge = self._get_random_edge()
            libsumo.edge.setAllowed(src_edge, "all")
            stage = libsumo.simulation.findRoute(src_edge, dst_edge, depart=depart)
            if stage.edges:
                break
        return src_edge, stage, dst_edge

    def _initialize_sliding_window(self):
        """Initializes the sliding window with the first positions of all users."""
        # Create random routes and vehicles
        cars = self._get_user_count(self.env.now, settings.NUMBER_OF_CARS)
        for _ in tqdm(range(cars), leave=False, desc="Creating random cars"):
            self._create_random_car(0)

        # Create random routes and people
        people = self._get_user_count(self.env.now, settings.NUMBER_OF_PEOPLE)
        for _ in tqdm(range(people), leave=False, desc="Creating random people"):
            self._create_random_person(0)

        # Create stationary users

        for _ in range(settings.ORCHESTRATOR_INTERVAL):
            window_slot = {}
            window_slot = self._get_slot_cars(window_slot)
            window_slot = self._get_slot_people(window_slot)
            window_slot = self._get_slot_stationary(window_slot)

            self.users_sliding_window.append(window_slot)
            libsumo.simulationStep()

    def _get_slot_cars(self, window_slot):
        """
        Get the positions of cars.
        Args:
            window_slot (dict): The current window slot with user positions.
        """
        for car in libsumo.vehicle.getIDList():
            lon, lat = libsumo.vehicle.getPosition(car)
            # UTM to WGS84
            lon, lat = libsumo.simulation.convertGeo(lon, lat)
            window_slot[car] = Location(lon, lat)
        return window_slot

    def _get_slot_people(self, window_slot):
        """
        Get the positions of people.
        Args:
            window_slot (dict): The current window slot with user positions.
        """
        for person in libsumo.person.getIDList():
            lon, lat = libsumo.person.getPosition(person)
            # UTM to WGS84
            lon, lat = libsumo.simulation.convertGeo(lon, lat)
            window_slot[person] = Location(lon, lat)
        return window_slot

    def _get_slot_stationary(self, window_slot):
        """
        Get the positions of stationary users.
        Args:
            window_slot (dict): The current window slot with user positions.
        """
        if len(self.users_sliding_window) == 0:
            for id in range(settings.NUMBER_OF_STATIONARY):
                window_slot[f"stationary_{id}"] = self._get_random_position()
        else:
            for id in range(settings.NUMBER_OF_STATIONARY):
                window_slot[f"stationary_{id}"] = self.users_sliding_window[-1][
                    f"stationary_{id}"
                ]
        return window_slot
