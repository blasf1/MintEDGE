import sys
import itertools
import logging
import math
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List
from mintedge import (
    Infrastructure,
    BaseStation,
    Service,
    Link,
)


class MintEDGEAllocationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AllocationStrategy:
    __slots__ = ["infr"]

    def __init__(self, infr: Infrastructure):
        """Algorithm responsible to allocate resources and assign
        computing requests to servers

        Args:
            infr (Infrastructure): Infrastructure to optimize
        """
        self.infr = infr

    def get_allocation(self, demand_matrix: Dict[str, Dict[str, int]]):
        """Allocate resources on the infrastructure in a greedy manner
        (closer first)

        Args:
            demand_matrix (Dict[str, Dict[str, int]]): Demand matrix

        Returns:
            Dict[str, int]: on/off state of the servers
            Dict[str, Dict[str, Dict[str, float]]]: Request assignation
            Dict[str, Dict[str, float]]: Resource allocation in servers
        """
        infr = self.infr

        # Initializations
        assig_matrix = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )  # type: Dict[str, Dict[str, Dict[str, float]]]

        alloc_matrix = defaultdict(
            lambda: defaultdict(float)
        )  # type: Dict[str, Dict[str, float]]

        server_status = {
            bs.name: 1 if bs.server is not None else 0 for bs in infr.bss.values()
        }  # all servers are active

        used_cap = {bs: 0 for bs in infr.bss}

        # Check if there is enough capacity in the infrastructure
        total_capacity = sum(
            bs.server.max_cap for bs in infr.bss.values() if bs.server is not None
        )
        total_demand = sum(
            serv.workload * demand_matrix[bs][serv.name]
            for bs in infr.bss
            for serv in infr.services.values()
        )

        if total_demand > total_capacity:
            raise MintEDGEAllocationError("Not enough capacity")

        # Main loop
        for src, serv in tqdm(
            itertools.product(infr.bss.values(), infr.services.values()),
            leave=False,
            desc="Allocating resources",
        ):
            # If there are no requests, skip
            if demand_matrix[src.name][serv.name] == 0:
                continue

            # Requests to route this iteration
            req_to_locate = demand_matrix[src.name][serv.name]

            # Get servers that can attend the requests within the constraints
            cand = self._get_cand_servers(server_status, src, serv)
            # Reroute requests
            assig_matrix, req_to_locate, used_cap, _ = self._route(
                cand,
                src,
                serv,
                req_to_locate,
                demand_matrix,
                assig_matrix,
                used_cap,
            )

        alloc_matrix = self._calculate_cpu_alloc_matrix(
            demand_matrix, assig_matrix, infr
        )

        return server_status, assig_matrix, alloc_matrix

    def _route(
        self,
        candidates: List[BaseStation],
        src: BaseStation,
        serv: Service,
        req_to_route: int,
        demand_mat: Dict[str, Dict[str, int]],
        assig_mat: Dict[str, Dict[str, Dict[str, float]]],
        used_cap: Dict[str, int],
    ):
        """Routes requests for service a coming from src to the servers in
        candidates depending on their capacity and that of the links.

        Args:
            candidates (List[BaseStation]): List of servers to route to
            src (BaseStation): Source server
            serv (Service): Service to which the requests belong
            req_to_route (int): Number of requests to route
            demand_mat (Dict[str, Dict[str, int]]): Request matrix
            assig_mat (Dict[str, Dict[str, Dict[str, float]]]): Current
                assignation matrix
            used_cap (Dict[str, int]): Used capacity in each server

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: New assignation
                matrix
            int: Number of requests not routed
            Dict[str, float]: Updated used capacity in each server
            float: Energy consumed in with the new assignation
        """
        e_route = 0
        for dst in tqdm(candidates, leave=False, desc="Assigning requests"):
            if req_to_route == 0:
                break
            path = self.infr.paths[(src.name, dst.name)]
            alpha = self._calculate_alpha(demand_mat, assig_mat, path)
            data = self._req_to_bits(req_to_route, serv)
            assig_req = 0
            if data <= alpha:
                avail_cap = dst.server.max_cap - used_cap[dst.name]
                if req_to_route <= self._ops_to_req(avail_cap, serv):
                    assig_req = req_to_route
                    req_to_route = 0
                else:
                    assig_req = self._ops_to_req(avail_cap, serv)
                    req_to_route -= assig_req
            else:
                assig_req = self._bits_to_req(alpha, serv)
                avail_cap = dst.server.max_cap - used_cap[dst.name]
                if assig_req >= self._ops_to_req(avail_cap, serv):
                    assig_req = self._ops_to_req(avail_cap, serv)

                req_to_route -= assig_req
            assig_mat[src.name][serv.name][dst.name] += (
                assig_req / demand_mat[src.name][serv.name]
            )
            used_cap[dst.name] += self._req_to_ops(assig_req, serv)
            e_route += (self._req_to_ops(assig_req, serv) * dst.server.op_energy) + (
                self._req_to_bits(assig_req, serv) * self.infr.get_path_sigma(path)
            )

        return assig_mat, req_to_route, used_cap, e_route

    def _calculate_transport_delay(
        self,
        src: BaseStation,
        serv: Service,
        dst: BaseStation,
    ) -> float:
        """Calculate delay in the transport of the request to the server

        Args:
            src (BaseStation): The Base Station where the request was received.
            serv (Service): The service being requested.
            dst (BaseStation): The server where the request is being sent.

        Returns:
            float: The delay in the transport of the request to the server.
        """
        t_u = src.get_delay(serv.input_size)  # RAN delay
        t_d = src.get_delay(serv.output_size)  # RAN delay output
        t_r = self.infr.get_path_delay(src, dst, serv)  # Backhaul
        t_o = self.infr.get_path_out_delay(dst, src, serv)  # Backhaul output
        return t_u + t_r + t_d + t_o

    def _ops_to_req(self, ops: int, serv: Service) -> int:
        """Converts operations into number of requests

        Args:
            ops (int): the number of operations to convert
            service (Service): the service of the requests evaluated

        Return:
            int: the number of requests
        """
        return math.floor(ops / serv.workload)

    def _req_to_ops(self, req: int, serv: Service) -> int:
        """Converts requests into number of operations

        Args:
            req (int): the number of requests to convert
            service (Service): the service of the requests evaluated

        Return:
            int: the number of operations
        """
        return req * serv.workload

    def _req_to_bits(self, req: int, serv: Service) -> int:
        """Converts requests into number of virtual machines

        Args:
            req (int): the number of requests to convert
            service (Service): the service of the requests evaluated

        Return:
            int: the number of bits to be transmitted
        """
        return req * serv.input_size

    def _bits_to_req(self, bits: int, serv: Service) -> int:
        """Converts bits into number of requests

        Args:
            bits (int): the number of bits to convert
            service (Service): the service of the requests evaluated

        Return:
            int: the number of requests
        """
        return math.floor(bits / serv.input_size)

    def _get_cand_servers(
        self,
        server_status: Dict[str, int],
        src: BaseStation,
        serv: Service,
    ) -> List[BaseStation]:
        """Returns a list of candidate servers for the service ak in bsi

        Args:
            server_status (Dict[str, int]): Dictionary server name-status
                indicating whether a server is present and on (1) or
                not present/off (0)
            src (BaseStation): Base station where the requests are received
            serv (str): Service name

        Returns:
            List[BaseStation]: List of servers that can attend the requests.
        """
        cand_servers = []

        # Get all paths from bsi to all other base stations.
        cand_paths = {k: v for k, v in self.infr.paths.items() if k[0] is src.name}
        # Get the RAN delay
        t_u = src.get_delay(serv.input_size)
        # Get BSs with a server and within the delay budget
        for path in cand_paths:
            dst = self.infr.bss[path[1]]
            if dst.server is not None and server_status[dst.name] == 1:

                t_c = serv.workload / dst.server.max_cap
                t_r = self.infr.get_path_delay(src, dst, serv)
                rem_delay_budget = serv.max_delay - t_u - t_c

                if t_r < rem_delay_budget:
                    cand_servers += [dst]
        # Order the servers according to the sigma of their paths
        cand_servers = self.sort_servers_by_sigma(cand_servers, src, self.infr)

        return cand_servers

    def sort_servers_by_sigma(
        self, servers: List[BaseStation], src: BaseStation, infr: Infrastructure
    ) -> List[BaseStation]:
        """Sorts a list of BSs with a server according to the sigma of the
            path from src.

        Args:
            servers (List[BaseStation]): List of servers to sort
            src (BaseStation): Source base station from which the path starts
            infr (Infrastructure): The infrastructure

        Returns:
            List[BaseStation]: Sorted list of BSs (decreasing order)
        """
        return sorted(
            servers,
            key=lambda x: infr.get_path_sigma(infr.paths[(src.name, x.name)]),
        )

    def _calculate_alpha_link(
        self,
        link: Link,
        demand_mat: Dict[str, Dict[str, int]],
        assig_mat: Dict[str, Dict[str, Dict[str, float]]],
    ) -> int:
        """Get the used capacity of a link in number of operations per second

        Args:
            link (Link): The link to get the used capacity of
            demand_mat (Dict[str, Dict[str, int]]): The request arrival matrix
            assig_mat (Dict[str, Dict[str, Dict[str, float]]]): The resource
                allocation matrix

        Return:
            int: The used capacity of the link in bps
        """
        cap = link.capacity
        # friendly name for list of services
        srvcs = self.infr.services.values()
        for (src, dst), serv in itertools.product(self.infr.paths.keys(), srvcs):
            path = self.infr.paths[(src, dst)]
            if link in path:
                cap -= (
                    assig_mat[src][serv.name][dst]
                    * demand_mat[src][serv.name]
                    * serv.input_size
                )

        if round(cap, 3) < 0:
            raise ValueError(f"Link capacity is negative: {cap}")

        return int(cap)

    def _calculate_alpha(
        self,
        demand_mat: Dict[str, Dict[str, int]],
        assig_mat: Dict[str, Dict[str, Dict[str, float]]],
        path: List[Link],
    ) -> int:
        """Calculate alpha (remaining capacity) for a given path.

        Args:
            demand_mat (Dict[str, Dict[str, int]]): matrix with the number of
                requests per service and bs.
            assig_mat (Dict[str, Dict[str, Dict[str, float]]]): routing fractions
                of the requests.
            path (List[Link]): path for which alpha is calculated

        Returns:
            int: Remaining capacity in bps.
        """
        return min(
            (self._calculate_alpha_link(link, demand_mat, assig_mat) for link in path),
            default=math.inf,  # type: ignore
        )

    def _calculate_cpu_alloc_matrix(
        self,
        demand_mat: Dict[str, Dict[str, int]],
        assig_mat: Dict[str, Dict[str, Dict[str, float]]],
        infr: Infrastructure,
    ) -> Dict[str, Dict[str, float]]:
        """Calculates the service-server CPU allocation matrix that represents
            the computing resource allocation.

        Args:
            demand_mat (Dict[str, Dict[str, int]]): Requests per service
                and bs.
            assig_mat (Dict[str, Dict[str, Dict[str, float]]]):
                Requests routing.
            infr (Infrastructure): The infrastructure.

        Returns:
            Dict[str, Dict[str, float]]: The service-server CPU allocation matrix.
        """

        def calculate_dst_workload(demand_mat, assig_mat, dst):
            return sum(
                self._req_to_ops(
                    assig_mat[src][serv.name][dst.name] * demand_mat[src][serv.name],
                    serv,
                )
                for src in infr.bss
                for serv in infr.services.values()
            )

        def get_total_req(demand_mat, assig_mat, service, dst):
            return sum(
                assig_mat[b][service.name][dst.name] * demand_mat[b][service.name]
                for b in infr.bss
            )

        def get_max_trans_delay(assignation_matrix, service, dst):
            return max(
                self._calculate_transport_delay(infr.bss[bsi], service, dst)
                for bsi in assignation_matrix.keys()
                if assignation_matrix[bsi][service.name][dst.name] > 0
            )

        alloc_mat = defaultdict(
            lambda: defaultdict(float)
        )  # type: Dict[str, Dict[str, float]]

        for dst in infr.bss.values():
            if dst.server is None:
                continue

            dst_workload = calculate_dst_workload(demand_mat, assig_mat, dst)

            if round(dst_workload) > dst.server.max_cap:
                raise MintEDGEAllocationError(
                    f"{dst_workload} exceeds {dst.name} capacity {dst.server.max_cap}."
                )

            if dst_workload == 0:
                continue
            t = math.inf
            for serv in infr.services.values():
                total_req = get_total_req(demand_mat, assig_mat, serv, dst)

                # Calculate the initial allocation for each service to the destination.
                alloc_mat[serv.name][dst.name] = (
                    self._req_to_ops(total_req, serv) / dst.server.max_cap
                )

                if alloc_mat[serv.name][dst.name] > 0:
                    max_trans_delay = get_max_trans_delay(assig_mat, serv, dst)
                    t = serv.max_delay - max_trans_delay  # type: float

                    # Adjust the allocation based on the maximum transport delay and service workload.
                    alloc_mat[serv.name][dst.name] = max(
                        alloc_mat[serv.name][dst.name],
                        serv.workload / (t * dst.server.max_cap),
                    )

            total_cpu = sum(alloc_mat[serv][dst.name] for serv in infr.services)

            if total_cpu > 1:
                excess = total_cpu - 1
                services = [
                    serv
                    for serv in infr.services.values()
                    if alloc_mat[serv.name][dst.name] > 0
                ]
                division = excess / len(services)
                length = len(services)

                # Adjust allocation to not exceed the destination's server capacity.
                for i, serv in enumerate(services):
                    min_alloc = serv.workload / (t * dst.server.max_cap)
                    if alloc_mat[serv.name][dst.name] - division < min_alloc:
                        division += division / (length - i)
                        continue
                    alloc_mat[serv.name][dst.name] -= division
            else:
                remaining = 1 - total_cpu
                services = [
                    serv
                    for serv in infr.services.values()
                    if alloc_mat[serv.name][dst.name] > 0
                ]
                for serv in services:
                    min_alloc = serv.workload / (t * dst.server.max_cap)
                    diff = min_alloc - alloc_mat[serv.name][dst.name]
                    if alloc_mat[serv.name][dst.name] < min_alloc and diff < remaining:
                        alloc_mat[serv.name][dst.name] += diff
                        remaining -= diff
                    elif alloc_mat[serv.name][dst.name] < min_alloc:
                        alloc_mat[serv.name][dst.name] += remaining
                        remaining = 0

                if remaining > 0:
                    for serv in services:
                        alloc_mat[serv.name][dst.name] += remaining / len(services)

        return alloc_mat
