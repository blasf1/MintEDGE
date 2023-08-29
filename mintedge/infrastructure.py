import math
from typing import Dict, List, Optional

import networkx as nx
import simpy
from tqdm import tqdm

from mintedge import (
    EnergyAware,
    EnergyMeasurement,
    EnergyModelLink,
    EnergyModelServer,
    Location,
    Service,
    User,
)


class EdgeServer(EnergyAware):
    __slots__ = [
        "env",
        "name",
        "is_on",
        "max_cap",
        "allocated_ops",
        "used_ops",
        "idle_power",
        "max_power",
        "op_energy",
        "energy_model",
        "allocated_ops_bs_a",
        "used_ops_bs_a",
        "boot_time",
        "last_onoff_time",
    ]

    def __init__(
        self,
        env: simpy.Environment,
        name: str,
        max_cap: float,
        idle_power: int,
        max_power: Optional[int] = None,
        boot_time: Optional[int] = None,
    ):
        """This class represents an edge server in the infrastructure.

        Args:
            name (str): Friendly name of the edge server
            max_cap (float): Maximum processing capacity in ops per second
            idle_power (int): Power consumed by the edge server when it is idle
            max_power (int, optional): Power consumed at 100% utilization
        """
        self.env = env
        self.name = name
        self.is_on = True
        self.max_cap = max_cap
        self.allocated_ops = 0  # Initially 0 capacity is allocated
        self.used_ops = 0  # Initially 0 capacity is allocated
        self.idle_power = idle_power
        self.max_power = max_power
        self.op_energy = (max_power - idle_power) / max_cap
        self.energy_model = EnergyModelServer()
        self.energy_model.set_parent(self)
        self.allocated_ops_bs_a: Dict[BaseStation, Dict[Service, int]] = {}
        self.used_ops_bs_a: Dict[BaseStation, Dict[Service, int]] = {}
        self.boot_time = boot_time
        self.last_onoff_time = 0

    def __repr__(self):
        return self.name

    def turn_off(self):
        """Turns off the edge server."""
        self.allocated_ops = 0
        self.is_on = False
        self.last_onoff_time = self.env.now

    def turn_on(self):
        """Turns on the edge server."""
        self.is_on = True
        self.last_onoff_time = self.env.now

    def get_utilization(self) -> float:
        """Returns the utilization of the edge server in the range [0, 1].

        Returns:
            float: Utilization of the edge server.
        """
        return self.used_ops / self.max_cap

    def measure_energy(self) -> EnergyMeasurement:
        """Measures the energy consumed by the edge server at the current
        time slot.

        Returns:
            EnergyMeasurement: Energy consumed by the edge server.
        """
        try:
            return self.energy_model.measure()
        except AttributeError:
            return EnergyMeasurement(self.idle_power, 0)

    def get_delay(self, frac: float, serv: Service) -> float:
        """Returns the computing delay in milliseconds.

        Args:
            frac (float): Fraction of the total capacity that is assigned
                for the instance.
            serv (Service): Service for which the delay is being calculated.

        Returns:
            float: Computing delay in seconds.
        """
        try:
            return serv.workload / (frac * self.max_cap)
            # return a.workload / self.max_cap
        except ZeroDivisionError:
            return -math.inf

    def get_avail_cap_bs_serv(self, src: "BaseStation", serv: Service) -> int:
        """Get the available capacity in ops per second for the given service
        instance to attend the requests coming from the given source.

        Args:
            src (BaseStation): Base station that is requesting the service.
            serv (Service): Service instance.

        Returns:
            int: Available capacity in ops per second.
        """
        try:
            return (
                self.allocated_ops_bs_a[src.name][serv.name]
                - self.used_ops_bs_a[src.name][serv.name]
            )
        except KeyError:
            return self.allocated_ops_bs_a[src.name][serv.name]

    def use_ops(self, ops: int, src: "BaseStation", serv: Service):
        """Uses ops to attend the requests of a service instance in this
        edge server. The ops must be released manually using releas_ops
        once the requests have been attended.

        Args:
            ops (int): Number of ops to use.
            src (BaseStation): Base station that is requesting the service.
            serv (Service): Service that is being requested.
        """
        if ops + self.used_ops > self.max_cap:
            raise ValueError(
                f"Cannot use {ops} ops on server {self.name}. Only {self.max_cap - self.used_ops} are available"
            )
        self.used_ops += ops

        try:
            if (
                self.used_ops_bs_a[src.name][serv.name] + ops
                > self.allocated_ops_bs_a[src.name][serv.name]
            ):
                raise ValueError(
                    f"Cannot use {ops} ops on server {self}. Only {self.allocated_ops_bs_a[src.name][serv.name]} are allocated for {src.name},{serv.name}"
                )
            self.used_ops_bs_a[src.name][serv.name] += ops
        except KeyError:
            try:
                self.used_ops_bs_a[src.name][serv.name] = ops
            except KeyError:
                self.used_ops_bs_a[src.name] = {serv.name: ops}

    def release_ops(self, ops: int, src: "BaseStation", a: Service):
        """Releases the given ops from the service instance that attends
        the requests from src in this edge server.
        Args:
            ops (int): Number of ops to release.
            src (BaseStation): Base station that is requesting the service.
            a (Service): Service that is being requested.
        """
        new_ops = self.used_ops - ops
        if new_ops < 0:
            raise ValueError(
                f"Cannot release {ops} ops on server {self.name}. Only {self.used_ops} in use"
            )
        self.used_ops -= ops

        if self.used_ops_bs_a[src.name][a.name] - ops < 0:
            raise ValueError(
                f"Cannot release {ops} ops on server {self}. Only {self.used_ops_bs_a[src.name][a.name]} in use"
            )
        self.used_ops_bs_a[src.name][a.name] -= ops

    def set_allocated_ops(self, src: "BaseStation", a: Service, req: int):
        """Allocates capacity for each service instance. A service instance
        attends a single source. If requests to a service come from more than
        one source then another instance is used.
        Args:
            src (BaseStation): Base Station that originated the request
            a (Service): Service that is being requested
            req (int): Number of requests made by the base station
        """

        try:
            self.allocated_ops_bs_a[src.name][a.name] = math.floor(
                req * a.workload
            )
        except KeyError:
            try:
                self.allocated_ops_bs_a[src.name] = {
                    a.name: math.floor(req * a.workload)
                }
            except KeyError:
                self.allocated_ops_bs_a = {
                    src.name: {a.name: math.floor(req * a.workload)}
                }

        self.allocated_ops = sum(
            sum(a.values()) for a in self.allocated_ops_bs_a.values()
        )
        if math.floor(req * a.workload) > self.max_cap:
            raise Exception(
                f"Cannot allocate {math.floor(req * a.workload)} requests on server {self.name} for {src.name},{a.name}."
            )

        if req < 0:
            raise Exception(
                f"Cannot allocate resources for negative number of requests. Something's wrong somewhere else."
            )


class BaseStation:
    __slots__ = ["name", "rate", "location", "server", "users"]

    def __init__(
        self,
        name: str,
        rate: int,
        edge_server: EdgeServer,
        location: Location,
    ):
        """This class represents a Base Station.
        Args:
            name (str): str that identifies the BS.
            rate (int): Bandwidth provided by the wireless link in bps.
            edge_server (EdgeServer): Edge server connected to this BS.
            location (Location): The (x,y) coordinates of the edge server
        """
        self.name = name
        self.rate = rate
        self.location = location
        self.server = edge_server
        self.users = []

    def __repr__(self):
        return self.name

    def get_delay(self, input_size: int) -> float:
        """Returns the RAN delay of a request made in this Base Station in
        milliseconds.
        Args:
            input_size (int): The size of the request in bits.
        Return:
            float: The RAN delay of a request made in this Base Station in
                seconds.
        """
        return input_size / (self.rate * 1000 * 1000)

    def set_edge_server(self, edge_server: EdgeServer):
        """Sets the edge server connected to this BS.
        Args:
            edge_server (EdgeServer): The edge server to connect to.
        """
        self.server = edge_server


class Link(EnergyAware):
    __slots__ = [
        "src",
        "dst",
        "capacity",
        "allocated_capacity",
        "sigma",
        "used_capacity",
        "energy_model",
    ]

    def __init__(
        self, src: BaseStation, dst: BaseStation, cap: float, sigma: float
    ):
        """A network link connecting base stations

        Args:
            src (BaseStation): Source Base Station of the network link.
            dst (BaseStation): Target Base Station of the network link.
            cap (float): Bandwidth provided by the network link in bps.
            sigma (float): Power consumed per bit transmitted over this link.
        """
        self.src = src
        self.dst = dst
        self.capacity = cap
        self.allocated_capacity = 0  # Initially no capacity is allocated
        self.used_capacity = 0  # Initially no capacity is used
        self.sigma = sigma
        self.energy_model = EnergyModelLink(self.sigma)
        self.energy_model.set_parent(self)

    def measure_energy(self) -> EnergyMeasurement:
        """Returns the energy consumed by this link in the last time slot

        Returns:
            EnergyMeasurement: The energy consumed by this link in the last
                time slot
        """
        try:
            return self.energy_model.measure()
        except AttributeError:
            return EnergyMeasurement(0, 0)

    def get_delay(self, input_size: int) -> float:
        """Returns the delay of a request routed through this link in
        seconds.

        Args:
            input_size (int): The size of the request in bytes.

        Return:
            The delay of a request routed through this link in seconds.
        """
        return input_size / self.capacity

    def use_bps(self, bps: int):
        """Allocates bps bits in the link.

        Args:
            bps (int): The capacity to be allocated in bps.
        """
        new_capacity = self.used_capacity + bps
        if new_capacity > self.capacity:
            raise Exception(
                f"Link {self.src.name},{self.dst.name} capacity exceeded by {new_capacity - self.capacity}."
            )
        self.used_capacity += bps

    def release_bps(self, bps: int):
        """Rleases bps bits in the link.
        Args:
            bps (int): The capacity to be released in bps.
        """
        new_capacity = self.used_capacity - bps
        if new_capacity < 0:
            raise Exception(
                f"Trying to release {bps} but only {self.used_capacity} is in use."
            )
        self.used_capacity -= bps

    def allocate_capacity(self, bps: int):
        """Allocates bps bits in the link.
        Args:
            bps (int): The capacity to be allocated in bps.
        """
        new_capacity = self.allocated_capacity + bps
        if new_capacity > self.capacity:
            raise Exception(
                f"Link {self.src.name},{self.dst.name} capacity exceeded. Capacity:{self.capacity}, Allocated:{self.allocated_capacity}, Requested:{bps}"
            )
        self.allocated_capacity = bps

    def check_capacity(self, bps: int):
        """Checks if bps bits can be allocated in the link.
        Args:
            bps (int): The capacity to be allocated in bps.
        """
        if self.used_capacity + bps > self.capacity:
            return False
        return True

    def __repr__(self):
        return f"({self.src.name},{self.dst.name})"


class Infrastructure:
    __slots__ = [
        "env",
        "bss",
        "services",
        "links",
        "paths",
        "assig_mat",
        "beta",
        "eta",
        "nxgraph",
        "kpis",
    ]

    def __init__(self, env: simpy.Environment):
        """Infrastructure graph of the simulated scenario.
        The infrastructure is multigraph G of BS and Links.
        Base Stations may or may not have Edge servers

        Args:
            env (simpy.Environment): The simulation environment.
        """
        self.env = env  # Simulation environment
        self.bss = {}  # Base Stations
        self.services = {}
        self.links = []
        self.paths = {}
        self.assig_mat = {}
        self.beta = {}
        self.eta = {}
        self.nxgraph = nx.Graph()
        self.kpis = {}

    def add_base_station(
        self,
        name: str,
        rate: int,
        edge_server: EdgeServer,
        location: Location,
    ):
        """Adds a Base Station to the infrastructure

        Args:
            name (str): Friendly name of the BS.
            rate (int): Bandwidth provided by the wireless link in bps.
            edge_server (EdgeServer): Edge server colocated in this BS.
            location (Location): The (x,y) coordinates of the BS.
        """
        self.bss[name] = BaseStation(name, rate, edge_server, location)
        self.nxgraph.add_node(
            self.bss[name], location=location, bs=self.bss[name]
        )

    def add_link(
        self, src: BaseStation, dst: BaseStation, capacity: int, sigma: float
    ):
        """Adds a link to the infrastructure

        Args:
            src (BaseStation): Source Base Station of the network link.
            dst (BaseStation): Target Base Station of the network link.
            capacity (int): Bandwidth provided by the network link in bps.
            sigma (float): Power consumed per bit transmitted over this link.
        """
        link = Link(src, dst, capacity, sigma)
        self.links.append(link)
        self.nxgraph.add_edge(src, dst, sigma=sigma, data=link)

    def add_service(self, service: Service):
        """Adds a service to the infrastructure

        Args:
            service (Service): Service to be added
        """
        self.services[service.name] = service

    def find_all_paths(self):
        """Finds all paths in the infrastructure. Call this function once the
        infrastructure has been created.
        """
        print("Finding all paths...")
        link_paths = {}
        paths = dict(nx.all_pairs_dijkstra_path(self.nxgraph, weight="sigma"))
        for src, dsts in tqdm(paths.items(), leave=False):
            for dst, path in dsts.items():
                link_paths[(src.name, dst.name)] = [
                    self.nxgraph.get_edge_data(path[i], path[i + 1])["data"]
                    for i in range(len(path) - 1)
                ]
        self.paths = link_paths

    def is_bs_isolated(self, bs: BaseStation) -> bool:
        """Checks if a server is isolated.

        Args:
            bs (BaseStation): Base Station that contains the server.
        Returns:
            True if the server is isolated, False otherwise.
        """
        return len(list(self.nxgraph.neighbors(bs))) == 0

    def get_path_sigma(self, path: List[Link]) -> float:
        """Returns the total energy consumed by each bit transmitted through
        a path.

        Args:
            path (List[Link]): List of links that form a path.
        Returns:
            The total energy consumed by each bit transmitted through a path
        """
        return sum(link.sigma for link in path)

    def update_backhaul_capacity(
        self,
        src: BaseStation,
        dst: BaseStation,
        a: Service,
        req: int,
    ):
        """Allocates capacity for req requests to service akin the backhaul
        links of the shortest path from src to dst.

        Args:
            src (BaseStation): Source Base Station.
            dst (BaseStation): Destination Base Station.
            ak (Service): Service that is being requested.
            req (int): Number of requests that need allocation.

        """
        bps = round(req * (a.input_size + a.output_size))
        if (src.name, dst.name) in self.paths:
            for link in self.paths[(src.name, dst.name)]:
                link.allocate_capacity(bps)

    def update_user_connection(self, user: User):
        """Updates the Base Stations in which each user is connected to their
        closest one.

        Args:
            user: User whose connection is to be updated.
        """
        connected = user.bs
        closest = min(
            self.bss.values(),
            key=lambda b: user.location.distance(b.location),
        )
        if connected != closest:
            user.bs.users.remove(user)
            user.bs = closest
            user.bs.users.append(user)

    def send_requests(
        self, env: simpy.Environment, src: BaseStation, serv: str, req: int
    ) -> List[simpy.Event]:
        """Receives requests from users and allocates capacity to them.

        Args:
            env (simpy.Environment): The simulation environment.
            src (BaseStation): Source Base Station.
            serv (str): Service that is being requested.
            req (int): Number of requests that need allocation.

        Returns:
            List[simpy.Event]: List of events that are triggered when the
                requests have been completely attended (and its capacity
                can be released)
        """

        if env.now not in self.kpis:
            self.kpis[env.now] = {}

        assig_mat = self.assig_mat[src.name][serv].copy()
        self._register_requests(env, src, serv, req)

        attended_req = 0
        events = []
        for dst in assig_mat:
            if assig_mat[dst] == 0:
                continue

            req_to_attend = round(req * assig_mat[dst])
            # Calculate rejections
            fitting_req, rejected = self._reject_requests(
                env, src, dst, serv, req_to_attend
            )

            if dst != src.name:
                # Allocate resources in the backhaul
                path = self.paths[(src.name, dst)]
                bps = math.ceil(
                    fitting_req
                    * (
                        self.services[serv].input_size
                        + self.services[serv].output_size
                    )
                )
                self._allocate_path_capacity(path, bps)

            # Allocate resources in the edge server
            self.bss[dst].server.use_ops(
                fitting_req * self.services[serv].workload,
                src,
                self.services[serv],
            )

            # create a process to eliminate the requests after they are attended
            events.append(
                env.process(
                    self._complete_req(
                        env, src, self.bss[dst], serv, fitting_req
                    )
                )
            )
            # Register servers' utilization
            self._register_server_utilization(env, dst)
            attended_req += fitting_req + rejected

        if attended_req < req - 1:  # missing one is fine. Rounding errors.
            src_a = f"{src}_{serv}"
            if f"rejected_req_{src_a}" in self.kpis[env.now]:
                self.kpis[env.now][f"rejected_req_{src_a}"] += (
                    req - attended_req
                )
            else:
                self.kpis[env.now][f"rejected_req_{src_a}"] = req - attended_req

            if "total_rejected" in self.kpis[env.now]:
                self.kpis[env.now]["total_rejected"] += req - attended_req
            else:
                self.kpis[env.now]["total_rejected"] = req - attended_req
        return events

    def _reject_requests(
        self,
        env: simpy.Environment,
        src: BaseStation,
        dst: str,
        serv: str,
        total_req: int,
    ):
        """Rejects requests that do not fit in the server or in the backhaul.

        Args:
            env (simpy.Environment): Simulation environment.
            src (BaseStation): Base Station where the requests were received.
            dst (str): Base Station where the server that attended the
                requests is.
            serv (str): Requested service's name.
            total_req (int): Total number of requests.

        Returns:
            Tuple[int, int]: Number of requests attended and
                number of rejected requests.
        """
        # Reject requests if they do not fit in the server
        rejected_s = 0
        avail_cap = self.bss[dst].server.get_avail_cap_bs_serv(
            src, self.services[serv]
        )
        fitting_req = math.floor(avail_cap / self.services[serv].workload)

        if fitting_req < total_req:
            rejected_s = total_req - fitting_req

        # Reject requests if they do not fit in the backhaul
        rejected_l = 0
        if dst != src.name:
            avail_cap = self.get_path_avail_cap(src, self.bss[dst])

            fitting_req_l = round(
                avail_cap
                / (
                    self.services[serv].input_size
                    + self.services[serv].output_size
                )
            )
            if fitting_req_l < fitting_req:
                rejected_l = fitting_req - fitting_req_l

        rejected = rejected_s + rejected_l
        self._register_rejections(env, src, serv, rejected)
        return total_req - rejected, rejected

    def _compute_delays(
        self, src: BaseStation, dst: BaseStation, a: str, req: int
    ):
        """Computes the delays of a batch of requests sent to the same BS at
        the same time and attended by the same edge server. The delays are
        added to the KPIs dictionary.

        Args:
            src (BaseStation): Source Base Station.
            dst (BaseStation): Destination Base Station.
            a (str): Service name to which the requests belong to.
            req (int): Number of requests.
        """
        if self.env.now not in self.kpis:
            self.kpis[self.env.now] = {}
        # Track delays
        t_u = src.get_delay(self.services[a].input_size)  # RAN delay
        # Backhaul delay
        if dst == src:
            t_r = 0
            t_o = 0
        else:
            t_r = self.get_path_delay(src, dst, self.services[a])
            t_o = self.get_path_out_delay(dst, src, self.services[a])

        t_c = dst.server.get_delay(
            self.beta[a][dst.name], self.services[a]
        )  # Compute delay
        t_d = dst.get_delay(self.services[a].output_size)  # RAN output delay
        delay = round(t_u + t_r + t_c + t_o + t_d, 5)

        if delay > self.services[a].max_delay:
            try:
                self.kpis[self.env.now]["unsatisf_req_" + a] += req
            except KeyError:
                self.kpis[self.env.now]["unsatisf_req_" + a] = req
        try:
            if delay > self.kpis[f"max_delay_{src.name}_{a}"]:
                self.kpis[self.env.now][f"max_delay_{src.name}_{a}"] = delay
        except KeyError:
            self.kpis[self.env.now][f"max_delay_{src.name}_{a}"] = delay

        try:
            self.kpis[self.env.now][f"delay_{src.name}_{a}"] += [delay] * req
        except KeyError:
            self.kpis[self.env.now][f"delay_{src.name}_{a}"] = [delay] * req

    def _complete_req(
        self,
        env: simpy.Environment,
        src: BaseStation,
        dst: BaseStation,
        a: str,
        req: int,
    ):
        """Releases capacity from requests from that have already been attended.

        Args:
            env (simpy.Environment): Simulation environment.
            src (BaseStation): Source base station (where the requests were
                received).
            dst (BaseStation): Destination base station (where the server
                that attended the requests is).
            a (str): Service name.
            req (int): Number of requests.
        """

        yield env.timeout(1)
        self._compute_delays(src, dst, a, req)
        # print(f"{env.now}: Completed {req} requests to {src.name}.")

        # Release resources in the edge server
        dst.server.release_ops(
            req * self.services[a].workload,
            src,
            self.services[a],
        )
        if dst != src:
            # Release resources in the backhaul
            path = self.paths[(src.name, dst.name)]
            bps = req * (
                self.services[a].input_size + self.services[a].output_size
            )
            self._release_path_capacity(path, bps)

    def _register_requests(
        self, env: simpy.Environment, src: BaseStation, serv: str, req: int
    ):
        """Register the total number of requests in this time step in
        the KPIs dictionary.

        Args:
            env (simpy.Environment): Simulation environment.
            src (BaseStation): Source base station (where the requests were
                received).
            serv (str): Service name.
            req (int): Number of requests.
        """
        # Register total requests
        if env.now not in self.kpis:
            self.kpis[env.now] = {}

        src_a = f"{src}_{serv}"
        if f"requests_{src_a}" in self.kpis[env.now]:
            self.kpis[env.now][f"requests_{src_a}"] += req
        else:
            self.kpis[env.now][f"requests_{src_a}"] = req

        if "total_requests" in self.kpis[env.now]:
            self.kpis[env.now]["total_requests"] += req
        else:
            self.kpis[env.now]["total_requests"] = req

    def _register_rejections(
        self, env: simpy.Environment, src: BaseStation, serv: str, rej: int
    ):
        """Register the number of rejected requests in the KPIs dictionary.

        Args:
            env (simpy.Environment): Simulation environment.
            src (BaseStation): Source base station (where the requests
                were received)
            serv (str): Service name.
            rej (int): Number of rejected requests.
        """
        if env.now not in self.kpis:
            self.kpis[env.now] = {}

        src_a = f"{src}_{serv}"
        if f"rejected_req_{src_a}" in self.kpis[env.now]:
            self.kpis[env.now][f"rejected_req_{src_a}"] += rej
        else:
            self.kpis[env.now][f"rejected_req_{src_a}"] = rej

        if "total_rejected" in self.kpis[env.now]:
            self.kpis[env.now]["total_rejected"] += rej
        else:
            self.kpis[env.now]["total_rejected"] = rej

    def _register_server_utilization(self, env: simpy.Environment, dst: str):
        """Save the current server utilization in the KPIs dictionary.

        Args:
            env (simpy.Environment): Simulation environment.
            dst (str): Name of the destination base station.
        """
        if self.bss[dst].server is not None:
            try:
                self.kpis[env.now]["server_util_" + dst] = round(
                    max(
                        self.bss[dst].server.get_utilization(),
                        self.kpis[env.now]["server_util_" + dst],
                    ),
                    4,
                )
            except KeyError:
                self.kpis[env.now]["server_util_" + dst] = round(
                    self.bss[dst].server.get_utilization(), 4
                )

    def _allocate_path_capacity(self, path: List[Link], bps: int):
        """Allocates capacity in all the links of a path in the backhaul.

        Args:
            path (List[Link]): List of links in the path.
            bps (int): Capacity to allocate.
        """
        for link in path:
            link.use_bps(bps)

    def _release_path_capacity(self, path: List[Link], bps: int):
        """Release capacity in all the links of a path in the backhaul.

        Args:
            path (List[Link]): List of links in the path.
            bps (int): Capacity to release.
        """
        for link in path:
            link.release_bps(bps)

    def get_path_delay(
        self, src: BaseStation, dst: BaseStation, a: Service
    ) -> float:
        """Returns the total delay of a path.

        Args:
            src (BaseStation): Source Base Station.
            dst (BaseStation): Destination Base Station.
            a (Service): Service that is being requested.

        Returns:
            float: The total delay of a path."""
        if src == dst:
            return 0
        return sum(
            link.get_delay(a.input_size)
            for link in self.paths[(src.name, dst.name)]
        )

    def get_path_out_delay(
        self, src: BaseStation, dst: BaseStation, a: Service
    ) -> float:
        """Returns the total delay of a path.

        Args:
            src (BaseStation): Source Base Station.
            dst (BaseStation): Destination Base Station.
            a (Service): Service that is being requested.

        Returns:
            float: The total delay of a path."""
        if src == dst:
            return 0
        path = self.paths[(src.name, dst.name)]
        return sum(link.get_delay(a.output_size) for link in path)

    def get_path_avail_cap(self, src: BaseStation, dst: BaseStation) -> int:
        """Returns the total capacity of a path.

        Args:
            src (BaseStation): Source Base Station.
            dst (BaseStation): Destination Base Station.

        Returns:
            float: The total capacity of a path."""
        if src == dst:
            return 0
        path = self.paths[(src.name, dst.name)]
        return min(link.capacity - link.used_capacity for link in path)
