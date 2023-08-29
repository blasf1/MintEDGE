import atexit
import itertools
import math
import os
import shutil
import statistics as stats
import tempfile
import uuid
from typing import Dict

import pysos
import simpy

import settings
from mintedge import (
    AllocationStrategy,
    EnergyMeter,
    IdealPredictor,
    Infrastructure,
    MobilityManager,
)


class Orchestrator:
    __slots__ = [
        "infr",
        "mobility_manager",
        "env",
        "assig_mat",
        "status_vec",
        "alloc_mat",
        "demand_mat",
        "max_lmbda",
        "predictor",
        "alloc_strategy",
        "measurements",
        "em_servers",
        "em_links",
    ]

    def __init__(
        self,
        infr: Infrastructure,
        mob_mngr: MobilityManager,
        env: simpy.Environment,
    ):
        """Orchestrator responsible for allocating resources on the infrastructure
        Args:
            infr (Infrastructure): The infrastructure managed by the orchestrator
            mob_mngr (MobilityManager): The mobility manager for access to user info
        """
        self.infr = infr
        self.mobility_manager = mob_mngr
        self.env = env
        self.assig_mat = self.initialize_assignation_matrix()
        self.status_vec = self.initialize_status_vector()
        self.alloc_mat = self.initialize_allocation_matrix()
        self.demand_mat = self.initialize_demand_matrix()
        self.predictor = IdealPredictor(infr, mob_mngr, env)
        self.alloc_strategy = AllocationStrategy(infr)

        # KPI related
        temp_dir = tempfile.mkdtemp()
        # Register a function to delete the temporary directory at exit
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        filename = uuid.uuid4().hex
        file_path = os.path.join(temp_dir, f"{filename}.json")
        self.measurements = pysos.Dict(file_path)
        self._initialize_measurements()

        # Energy meters
        self.em_servers = EnergyMeter(
            [bs.server for bs in infr.bss.values() if bs.server is not None],
            "servers",
            settings.ENERGY_MEASUREMENT_INTERVAL,
        )

        self.em_links = EnergyMeter(
            infr.links,
            "links",
            settings.ENERGY_MEASUREMENT_INTERVAL,
        )

        # Run energy meters processes
        env.process(self.em_servers.run(env))
        env.process(self.em_links.run(env))

    def run(self, env: simpy.Environment):
        """Starts the orchestrator process

        Args:
            env (simpy.Environment): The simulation environment
        """

        # Run orchestrator process
        while True:
            new_demand_mat = self._get_current_demand_matrix()
            self.demand_mat = new_demand_mat  # Update lmbda matrix

            # Update allocation matrixes
            if (
                env.now % settings.ORCHESTRATOR_INTERVAL == 0
                or env.now == 1
                or self._reaction_needed(env.now)
            ):
                # Get the maximum lambda for the next prediction interval
                if settings.USE_PREDICTOR:
                    new_demand_mat = self._get_max_pred_demand_matrix()

                # Apply capacity buffer if needed
                if settings.CAPACITY_BUFFER > 0:
                    new_demand_mat = self._apply_capacity_buffer(new_demand_mat)

                # If the demand is higher than the available resources, reject
                new_demand_mat = self._get_max_acceptable(new_demand_mat)

                # New Resource Allocation
                (
                    status_vec,
                    assig_mat,
                    alloc_mat,
                ) = self.alloc_strategy.get_allocation(new_demand_mat)

                # Place resources as computed by the allocation strategy
                self.allocate(new_demand_mat, status_vec, assig_mat, alloc_mat)
                # Update current allocation
                self.status_vec = status_vec
                self.assig_mat = assig_mat
                self.alloc_mat = alloc_mat

            # Gather and store kpis
            self.save_kpis(env.now)  # TODO: Make it more storage-efficient
            yield env.timeout(1)

    def _reaction_needed(self, time: int):
        # Reactive allocation
        share_of_rejections = 0
        if self.measurements[time - 1]["total_requests"] > 0:
            share_of_rejections = (
                self.measurements[time - 1]["total_rejected"]
                / self.measurements[time - 1]["total_requests"]
            )
        if (
            settings.REACTIVE_ALLOCATION
            and share_of_rejections > settings.REACTION_THRESHOLD
        ):
            return True
        else:
            return False

    def _apply_capacity_buffer(self, demand_mat: Dict[str, Dict[str, int]]):
        """Apply a capacity buffer to the demand matrix so that the allocation
        strategy allocates extra space for future requests.

        Args:
            demand_mat (Dict[str, Dict[str, int]]): The request matrix to
                apply the buffer to.
        """

        for bs in demand_mat:
            for serv in demand_mat[bs]:
                demand_mat[bs][serv] = int(
                    demand_mat[bs][serv] * (1 + settings.CAPACITY_BUFFER)
                )
        return demand_mat

    def _get_max_acceptable(self, demand_mat: Dict[str, Dict[str, int]]):
        """Rejects not fitting requests

        Args:
            demand_mat (Dict[str, Dict[str, int]]): Request matrix

        Returns:
            Dict[str, Dict[str, int]]: Request matrix after rejecting requests
        """
        bss = self.infr.bss.values()
        servcs = self.infr.services.values()

        available = sum(
            bs.server.max_cap for bs in bss if bs.server is not None
        )
        required = sum(
            demand_mat[bs.name][serv.name] * serv.workload
            for bs in bss
            for serv in servcs
        )
        if required > available:
            reject_factor = available / required
            for bs in demand_mat:
                for serv in demand_mat[bs]:
                    demand_mat[bs][serv] = int(
                        demand_mat[bs][serv] * reject_factor
                    )

        return demand_mat

    def _get_current_demand_matrix(self) -> Dict[str, Dict[str, int]]:
        """Updates the current lmbda matrix with user mobility information.
        Returns:
            Dict[str, Dict[str, int]]: The updated lmbda matrix
        """
        new_demand_mat = self.initialize_demand_matrix()
        bss = self.infr.bss.values()
        for bs, serv in itertools.product(bss, self.infr.services):
            for user in bs.users:
                new_demand_mat[bs.name][serv] += user.lmbda[serv]

        return new_demand_mat

    def _get_max_pred_demand_matrix(self) -> Dict[str, Dict[str, int]]:
        """Returns the maximum lambda for a given period.

        Returns:
            Dict[str, Dict[str, int]]: The maximum lambda for a given period.
        """

        return self.predictor.get_max_demand()

    def allocate(
        self,
        new_demand_mat: Dict[str, Dict[str, int]],
        new_status_vec: Dict[str, int],
        new_assig_mat: Dict[str, Dict[str, Dict[str, float]]],
        new_alloc_mat: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """Allocate resources on the infrastructure according to the
        allocation matrixes.

        Args:
            new_demand_mat (Dict[str, Dict[str, int]]): New demand matrix
            new_status_vec (Dict[str, int]): New status vector
            new_assig_mat (Dict[str, Dict[str, Dict[str, float]]]):
                New assignation matrix
            new_alloc_mat (Dict[str, Dict[str, Dict[str, float]]]):
                New allocation matrix
        """
        self.infr.assig_mat = new_assig_mat
        self.infr.beta = new_alloc_mat
        self.infr.eta = new_status_vec

        bss = self.infr.bss.values()
        servs = self.infr.services.values()

        self._apply_status_vector(new_status_vec)
        for dst in self.infr.bss.values():
            if dst.server is not None and dst.server.is_on:
                for src, serv in itertools.product(bss, servs):
                    # Apply allocation
                    dst.server.set_allocated_ops(
                        src,
                        serv,
                        new_demand_mat[src.name][serv.name]
                        * new_assig_mat[src.name][serv.name][dst.name],
                    )
                    # Apply assigantion
                    if src != dst:
                        self.infr.update_backhaul_capacity(
                            src,
                            dst,
                            serv,
                            new_demand_mat[src.name][serv.name]
                            * new_assig_mat[src.name][serv.name][dst.name],
                        )

    def _apply_status_vector(self, new_status_vec: Dict[str, int]):
        """Apply the new status vector to the infrastructure

        Args:
            new_status_vec (Dict[str, int]): the status vector
        """
        for dst in self.infr.bss.values():
            if (
                new_status_vec[dst.name] == 0
                and dst.server is not None
                and dst.server.is_on
            ):
                dst.server.turn_off()
            elif (
                new_status_vec[dst.name] == 1
                and dst.server is not None
                and not dst.server.is_on
            ):
                dst.server.turn_on()

    def save_kpis(self, time: int):
        """Save KPIs of the current iteration in the measurements dataframe

        Args:
            time (int): Current time
        """
        if time == 0:
            return  # Nothing to save at time 0 (nothing has happened yet)

        # Initialize row with 0s
        new_row = {k: 0 for k in self.measurements[0].keys()}

        # Get KPIs from mobility manager
        new_row["active_users"] = self.mobility_manager.get_running_user_count()

        # Get KPIs from energy meters
        new_row["dynamic_W_servers"] = round(
            self.em_servers.measurmnts[-1].dynamic, 3
        )
        new_row["idle_W_servers"] = round(
            self.em_servers.measurmnts[-1].idle, 3
        )
        new_row["W_links"] = round(self.em_links.measurmnts[-1].dynamic, 3)

        # Get KPIs from infrastructure
        try:
            k, v = list(self.infr.kpis.items())[-1]
            for k, v in v.items():
                if isinstance(v, list):
                    try:
                        mean = round(stats.mean(v), 5)
                        new_row[k] = None if math.isinf(mean) else mean
                    except stats.StatisticsError:
                        new_row[k] = 0
                else:
                    new_row[k] = None if math.isinf(v) else v
        except IndexError:
            pass

        self.measurements[self.env.now] = new_row  # Store in disk
        self.infr.kpis = {}  # Reset KPIs for next iteration

    def initialize_assignation_matrix(
        self,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Initialize the gamma matrix to 0
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: The initialized gamma matrix
        """
        bss = self.infr.bss
        servcs = self.infr.services
        gamma = {
            bsi: {ak: {bsj: 0 for bsj in bss} for ak in servcs} for bsi in bss
        }
        return gamma

    def initialize_status_vector(self) -> Dict[str, int]:
        """Initialize the eta matrix
        Returns:
            Dict[str, int]: The initialized eta vector
        """
        eta = {}
        return eta

    def initialize_allocation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize the beta matrix to 0
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: The initialized beta matrix
        """
        bss = self.infr.bss
        servcs = self.infr.services
        beta = {ak: {bsj: 0 for bsj in bss} for ak in servcs}

        return beta

    def initialize_demand_matrix(self) -> Dict[str, Dict[str, int]]:
        """Initialize the lmbda matrix to 0
        Returns:
            Dict[str, Dict[str, int]]: The initialized lmbda matrix
        """
        return {
            bs: {ak: 0 for ak in self.infr.services} for bs in self.infr.bss
        }

    def _initialize_measurements(self):
        """Initialize the measurements dictionary"""
        services = self.infr.services
        bss = self.infr.bss

        kpis = (
            [
                "active_users",
                "dynamic_W_servers",
                "idle_W_servers",
                "W_links",
                "total_requests",
                "total_rejected",
            ]
            + [f"requests_{bs}_{serv}" for serv in services for bs in bss]
            + [f"rejected_req_{bs}_{serv}" for serv in services for bs in bss]
            + [f"unsatisf_req_{serv}" for serv in services]
            + [f"max_delay_{src}_{serv}" for src in bss for serv in services]
            + [f"delay_{src}_{serv}" for src in bss for serv in services]
            + [
                f"server_util_{bs}"
                for bs in bss
                if self.infr.bss[bs].server is not None
            ]
        )

        init = {k: 0 for k in kpis}
        self.measurements[0] = init
