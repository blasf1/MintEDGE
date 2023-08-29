from abc import ABC, abstractmethod
from typing import Dict

import libsumo
import simpy

import settings
from mintedge import Service

_users_created = 0


class User(ABC):
    """This class represents a User.
    Args:
        name: str that identifies the User.
        location: The (x,y) coordinates of the user.
    """

    __slots__ = [
        "env",
        "id",
        "mob_mngr",
        "location",
        "bs",
        "lmbda",
        "start_time",
    ]

    def __init__(
        self,
        env: simpy.Environment,
        id: str,
        infr: "Infrastructure",
        mob_mngr: "MobilityManager",
        loc: "Location",
        start_time: float,
    ):
        global _users_created
        _users_created += 1
        self.env = env
        self.infr = infr
        self.mob_mngr = mob_mngr
        if id is not None:
            self.id = id
        else:
            self.id = f"User {_users_created}"
        # The (x,y) coordinates of the user
        self.location = loc
        # The Base Station the user is connected to
        self.bs = self.get_closest_bs()
        self.lmbda = self._set_lmbda()

        self.start_time = start_time

    @abstractmethod
    def _set_lmbda(self) -> Dict["Service", int]:
        """Set the lambda for each service.
        Returns:
            A dictionary with the number of requests for each service
            per unit of time.
        """

    def run(self, env):
        from mintedge import Location, RAND_NUM_GEN

        while env.now < self.start_time:
            yield env.timeout(1)

        self.bs.users.append(self)
        ends = []
        while True:
            # Update my position
            try:
                loc = self.mob_mngr.get_user_location(self.id)
            except KeyError:
                break  # If the car has finished break the loop and finish
            self.location = loc

            yield simpy.events.AllOf(env, ends)
            ends = []
            # if env.now % settings.MOBILITY_STEP == 0:
            self.infr.update_user_connection(self)

            # Send requests
            for a in self.lmbda:
                if self.lmbda[a] > 0:
                    ends += self.infr.send_requests(
                        env, self.bs, a, RAND_NUM_GEN.poisson(self.lmbda[a])
                    )
            if not ends:
                yield env.timeout(1)

        # It's deadtime. Kill it
        self.bs.users.remove(self)

    def set_user_location(self, loc: "Location"):
        self.location = loc

    def get_closest_bs(self):
        return min(
            self.infr.bss.values(),
            key=lambda b: self.location.distance(b.location),
        )

    def __repr__(self):
        return self.name


class Car(User):
    def __init__(
        self,
        env: simpy.Environment,
        id: str,
        infr: "Infrastructure",
        mob_mngr: "MobilityManager",
        loc: "Location",
        start_time: float,
    ):
        super().__init__(env, id, infr, mob_mngr, loc, start_time)

    def _set_lmbda(self) -> Dict[Service, int]:
        return {
            a.name: a.arrival_rate if a.name in settings.CAR_SERVICES else 0
            for a in settings.SERVICES
        }


class Person(User):
    def __init__(
        self,
        env: simpy.Environment,
        id: str,
        infr: "Infrastructure",
        mob_mngr: "MobilityManager",
        loc: "Location",
        start_time: float,
    ):
        super().__init__(env, id, infr, mob_mngr, loc, start_time)

    def _set_lmbda(self) -> Dict[Service, int]:
        return {
            a.name: a.arrival_rate
            if a.name in settings.PEDESTRIAN_SERVICES
            else 0
            for a in settings.SERVICES
        }


class Stationary(User):
    def __init__(
        self,
        env: simpy.Environment,
        id: str,
        infr: "Infrastructure",
        mob_mngr: "MobilityManager",
        loc: "Location",
        start_time: float,
    ):
        super().__init__(env, id, infr, mob_mngr, loc, start_time)
        self.speed = 0

    def _set_lmbda(self) -> Dict[Service, int]:
        return {
            a.name: a.arrival_rate
            if a.name in settings.STATIONARY_SERVICES
            else 0
            for a in settings.SERVICES
        }
