from typing import Dict
from simpy.core import Environment
import settings
import itertools
from mintedge import Infrastructure, BaseStation, MobilityManager, Location


class IdealPredictor:
    __slots__ = ["infr", "mob_mngr", "env"]

    def __init__(
        self, infr: Infrastructure, mob_mngr: MobilityManager, env: Environment
    ):
        """This class represents an ideal predictor.
        It can be replaced by any other kind of load predictor.

        Args:
            infr (Infrastructure): the infrastructure of the simulation.
            mob_mngr (MobilityManager): the mobility manager of the simulation.
            env (Environment): the simulation environment.
        """
        self.infr = infr
        self.mob_mngr = mob_mngr
        self.env = env

    def get_max_demand(self) -> Dict[str, Dict[str, int]]:
        """Returns the maximum lambda for a given period.

        Returns:
            Dict[str, Dict[str, float]]: The maximum lambda for a given period.
        """
        max_demand = {bs: {a: 0 for a in self.infr.services} for bs in self.infr.bss}

        users_per_time = self.mob_mngr.users_sliding_window
        services = self.infr.services
        bss = self.infr.bss

        for users in users_per_time:
            new_demand = {
                bs: {a: 0 for a in self.infr.services} for bs in self.infr.bss
            }
            for _, loc in users.items():
                bs = self.get_connected_bs(loc)
                if bs is None:
                    continue
                for serv in settings.SERVICES:
                    # TODO: This works only if all users use all services
                    max_demand[bs.name][serv.name] += serv.arrival_rate  # type: ignore
            for bs, serv in itertools.product(bss, services):
                max_demand[bs][serv] = max(max_demand[bs][serv], new_demand[bs][serv])

        return max_demand

    def get_connected_bs(self, loc: Location) -> BaseStation:
        """Returns the Base Station to which a user in loc would be connected
        to is connected to.

        Args:
            loc (Location): Location of the user.

        Returns:
            BaseStation: Base Station in which the user will be connected.
        """

        return min(
            self.infr.bss.values(),
            key=lambda b: loc.distance(b.location),
        )
