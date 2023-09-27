import atexit
import itertools
import os
import shutil
import tempfile
import uuid
import requests
import tempfile
import libsumo
import sumolib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import simpy
from tqdm.auto import tqdm

import mintedge
import settings


class MintEDGESettingsError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MintEDGEError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class Simulation:
    __slots__ = ["simulation_time", "seed", "output_file"]

    def __init__(self, simulation_time: float, output_file: str, seed: int):
        self.simulation_time = simulation_time
        self._set_simulation_time(simulation_time)
        self.seed = seed
        self._set_seed(seed)
        self.output_file = output_file
        self._check_settings()
        try:
            if (
                settings.NORTH is not None
                and settings.SOUTH is not None
                and settings.EAST is not None
                and settings.WEST is not None
            ):
                self.create_sumo_net(
                    settings.NORTH, settings.SOUTH, settings.EAST, settings.WEST
                )
        except AttributeError:
            pass

    def run(self):
        #  Set up simulation
        env = simpy.Environment()
        if settings.RANDOM_ROUTES:
            mobility_manager = mintedge.RandomMobilityManager(env)
        else:
            mobility_manager = mintedge.MobilityManager(env)
        infr = self.create_infrastructure(env)
        if settings.PLOT_SCENARIO:
            self.plot_scenario(infr)
        orch = mintedge.Orchestrator(infr, mobility_manager, env)
        env.process(mobility_manager.run(env, infr))
        env.process(orch.run(env))

        #  Run simulation
        print("Running simulation...")
        for until in tqdm(range(1, self.simulation_time + 1)):
            env.run(until=until)

        #  Write results to file
        df = pd.DataFrame.from_dict(orch.measurements, orient="index")
        df.to_csv(self.output_file, index_label="time", sep=",")

    def _set_seed(self, seed):
        mintedge.SEED = seed
        mintedge.RAND_NUM_GEN = np.random.default_rng(seed=seed)

    def _set_simulation_time(self, sim_time):
        mintedge.SIMULATION_TIME = sim_time

    def plot_scenario(self, infr):
        def plot_roads(net, ax):
            from matplotlib.collections import LineCollection

            shapes = []
            for e in net._edges:
                shapes.append(e.getShape())

            line_segments = LineCollection(
                shapes, linewidths=0.01, colors="#000000", zorder=-1
            )
            ax.add_collection(line_segments)

        def plot_network(net):
            pos = {}
            node_color = []
            for node in infr.nxgraph.nodes:
                x, y = net.convertLonLat2XY(
                    infr.nxgraph.nodes[node]["location"].x,
                    infr.nxgraph.nodes[node]["location"].y,
                )
                pos[node] = (x, y)
                # Determine node color based on the presence of a server
                if infr.nxgraph.nodes[node]["bs"].server is not None:
                    node_color.append("#0000FF")
                else:
                    # Green color for nodes without a server
                    node_color.append("#FF0000")

            nx.draw(
                infr.nxgraph,
                pos=pos,
                with_labels=False,
                node_size=30,
                width=3,
                font_size=5,
                edge_color="#ff6c6c",
                node_color=node_color,
                ax=ax,
            )

        net = sumolib.net.readNet(settings.NET_FILE)
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_roads(net, ax)
        plot_network(net)
        # Show the plot
        plt.savefig("network_graph.pdf", bbox_inches="tight")

    def create_sumo_net(
        self, north: float, south: float, east: float, west: float
    ):
        print("Net file not provided, generating from OSM")
        from distutils.spawn import find_executable

        if find_executable("netconvert") is None:
            raise MintEDGEError("netconvert is not installed")

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        # Register a function to delete the temporary directory at exit
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

        output_file_path = os.path.join(temp_dir, "map.net.xml")

        # Define the URL and parameters for the download
        url = f"http://overpass.openstreetmap.ru/cgi/xapi_meta?*[bbox={west},{south},{east},{north}]"
        # Get the map from OSM and save it to a temporary file
        print("Downloading map from OSM, this may take a while...")
        response = requests.get(url)

        if response.status_code == 200:
            # Write the response content to the temporary file
            with tempfile.NamedTemporaryFile(
                dir=temp_dir, suffix=".osm.xml", delete=False
            ) as temp_file:
                temp_file.write(response.content)

                # Convert the OSM file to a SUMO network file
                call = f"netconvert --osm-files {temp_file.name} -o {output_file_path} --no-warnings --ignore-errors --remove-edges.isolated --remove-edges.by-type railway.rail,railway.tram,railway.light_rail,railway.subway,railway.preserved,highway.pedestrian,highway.cycleway,highway.footway,highway.bridleway,highway.steps,highway.step,highway.stairs --ramps.guess --junctions.join --tls.join --no-internal-links --no-turnarounds --roundabouts.guess --offset.disable-normalization --output.original-names"
                os.system(call)
        else:
            raise MintEDGEError(
                f"Failed to download file. HTTP status code: {response.status_code}"
            )
        settings.NET_FILE = output_file_path

    def _filter_infrastructure(self, df_bss, df_links):
        (w, s), (e, n) = libsumo.simulation.getNetBoundary()
        w, s = libsumo.simulation.convertGeo(w, s)
        e, n = libsumo.simulation.convertGeo(e, n)
        print(
            f"Map boundaries: west - {w}, south - {s}, east - {e}, north - {n}"
        )
        df_bss = df_bss[(df_bss["provider"] == settings.PROVIDER)]
        df_bss = df_bss[(df_bss["lon"] > w)]
        df_bss = df_bss[(df_bss["lon"] < e)]
        df_bss = df_bss[(df_bss["lat"] > s)]
        df_bss = df_bss[(df_bss["lat"] < n)]
        df_bss = df_bss.reset_index(drop=True)

        # TODO: This won't work when the file is not empty
        if not df_links.empty:
            df_links = df_links[
                (df_links["src_id"].isin(df_bss["location_id"]))
                & (df_links["dst_id"].isin(df_bss["location_id"]))
            ]

        return df_bss, df_links

    def make_connected(self, infr):
        G = infr.nxgraph

        # Create a list of connected components in the graph
        connected_comp = list(nx.connected_components(G))

        # If there is only one connected component, the graph is already connected
        if len(connected_comp) == 1:
            return infr
        # Otherwise, connect the connected components by adding edges between the closest nodes
        closest_comp1, closest_comp2, min_dist = None, None, float("inf")
        for i in range(len(connected_comp) - 1):
            comp1 = list(connected_comp[i])
            for j in range(i + 1, len(connected_comp)):
                comp2 = list(connected_comp[j])
                for node1, node2 in itertools.product(comp1, comp2):
                    distance = node1.location.distance(node2.location)
                    if distance < min_dist:
                        closest_comp1, closest_comp2, min_dist = (
                            comp1,
                            comp2,
                            distance,
                        )

        closest_node1, closest_node2, min_dist = None, None, float("inf")
        for node1, node2 in itertools.product(closest_comp1, closest_comp2):
            distance = node1.location.distance(node2.location)
            if distance < min_dist:
                closest_node1, closest_node2, min_dist = (
                    node1,
                    node2,
                    distance,
                )

        infr.add_link(
            infr.bss[closest_node1.name],
            infr.bss[closest_node2.name],
            settings.MAX_LINK_CAPACITY,
            settings.W_PER_BIT,
        )

        # Check if there are still isolated subgraphs and recursively connect them
        return self.make_connected(infr)

    def top_k_server_placement(self, env, infr, k):
        # Sort the nodes by their degree
        nodes = sorted(infr.nxgraph.degree, key=lambda x: x[1], reverse=True)

        # Place the servers
        for i in range(k):
            node_name = nodes[i][0].name
            infr.bss[node_name].set_edge_server(
                mintedge.EdgeServer(
                    env,
                    node_name,
                    settings.MAX_SERVER_CAPACITY,
                    idle_power=settings.IDLE_SERVER_POWER,
                    max_power=settings.MAX_SERVER_POWER,
                    boot_time=settings.SERVER_BOOT_TIME,
                )
            )

        return infr

    def deterministic_server_placement(self, env, infr, k):
        # Calculate the centrality of each node based on degree and location
        centrality = {}
        for node in infr.nxgraph.nodes:
            # Degree centrality
            centrality[node] = infr.nxgraph.degree[node] / len(infr.nxgraph)

            # Location-based centrality
            x = infr.nxgraph.nodes[node]["location"].x
            y = infr.nxgraph.nodes[node]["location"].y
            distance_sum = 0
            for neighbor in infr.nxgraph.neighbors(node):
                x_neighbor = infr.nxgraph.nodes[neighbor]["location"].x
                y_neighbor = infr.nxgraph.nodes[neighbor]["location"].y
                distance = (
                    (x - x_neighbor) ** 2 + (y - y_neighbor) ** 2
                ) ** 0.5
                distance_sum += distance
            # Multiply by inverse distance sum to give more weight to nodes
            # that are further away from their neighbors (spread them through the map)
            centrality[node] *= 1 / (distance_sum + 1)

        # Sort the nodes by their centrality in descending order
        nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        # Number of server types
        num_server_types = len(settings.SERVERS)

        # Place the servers
        for i in range(min(k, len(nodes))):
            node = nodes[i][0]
            # Get server settings
            ser_set = settings.SERVERS[i % num_server_types]
            node.set_edge_server(
                mintedge.EdgeServer(
                    env,
                    node.name,
                    ser_set["MAX_CAPACITY"],
                    idle_power=ser_set["IDLE_POWER"],
                    max_power=ser_set["MAX_POWER"],
                    boot_time=ser_set["BOOT_TIME"],
                )
            )

        return infr

    def create_infrastructure(
        self, env: simpy.Environment
    ) -> mintedge.Infrastructure:
        infr = mintedge.Infrastructure(env)
        # Add services
        for s in settings.SERVICES:
            infr.add_service(s)

        df = pd.read_csv(settings.BSS_FILE)

        try:
            df_links = pd.read_csv(settings.LINKS_FILE)
        except pd.errors.EmptyDataError:
            df_links = pd.DataFrame()

        df, df_links = self._filter_infrastructure(df, df_links)
        # If the area does not have any base stations, stop the simulation
        if len(df) == 0:
            raise MintEDGEError("No base stations found in the given area")

        for i, row in tqdm(
            df.iterrows(), desc="Adding base stations", leave=False
        ):
            location = mintedge.Location(row["lon"], row["lat"])
            infr.add_base_station(
                "BS" + str(i), settings.BS_DATARATE, None, location
            )  # Servers (None) added later

        print("Number of base stations imported: ", len(infr.bss))
        link_dict = {}
        for i, row in df_links.iterrows():
            try:
                site1 = "BS" + str(
                    df[df["location_id"] == row["src_id"]].index[0]
                )
                site2 = "BS" + str(
                    df[df["location_id"] == row["dst_id"]].index[0]
                )
            except IndexError:
                continue
            col = "r"
            link_dict[(site1, site2)] = col
            link_dict[(site2, site1)] = col

        for (site1, site2), _ in tqdm(
            link_dict.items(), leave=False, desc="Adding links"
        ):
            infr.add_link(
                infr.bss[site1],
                infr.bss[site2],
                settings.MAX_LINK_CAPACITY,
                settings.W_PER_BIT,
            )

        # Make the graph connected
        infr = self.make_connected(infr)

        # Place the servers
        k = int(len(infr.bss) * settings.SHARE_OF_SERVERS)

        infr = self.deterministic_server_placement(env, infr, k)

        infr.find_all_paths()
        del df
        del df_links
        return infr

    def _check_settings(self):
        # Check that there is a BSS file. Random BSSs is not yet supported
        if settings.BSS_FILE is None or not os.path.isfile(settings.BSS_FILE):
            raise MintEDGESettingsError("Currently, BSS_FILE must be set")

        # Random links is supported but, atm, file is needed even if empty
        if settings.LINKS_FILE is None or not os.path.isfile(
            settings.LINKS_FILE
        ):
            raise MintEDGESettingsError("Currently, LINKS_FILE must be set")

        # If random routes are not used, a routes file must be provided
        if not settings.RANDOM_ROUTES and settings.ROUTES_FILE is None:
            raise MintEDGESettingsError(
                "If RANDOM_ROUTES is False, ROUTES_FILE must be set"
            )

        # If a routes file is provided, it must be a valid file
        if (
            settings.ROUTES_FILE is not None
            and not os.path.isfile(settings.ROUTES_FILE)
            and not settings.RANDOM_ROUTES
        ):
            raise MintEDGESettingsError("ROUTES_FILE is not a valid file")

        # If random routes are used, the number of cars must be set
        if settings.RANDOM_ROUTES and settings.NUMBER_OF_CARS is None:
            raise MintEDGESettingsError(
                "If RANDOM_ROUTES is True, NUMBER_OF_USERS must be set"
            )

        if settings.ROUTES_FILE is not None and settings.NET_FILE is None:
            raise MintEDGESettingsError(
                "If ROUTES_FILE is set, NET_FILE must be set"
            )

        # Either the net file or the bounds must be set
        try:
            if (
                settings.NORTH is None
                or settings.SOUTH is None
                or settings.EAST is None
                or settings.WEST is None
            ) and settings.NET_FILE is None:
                raise MintEDGESettingsError(
                    "Either NET_FILE or all bounds (NORTH, SOUTH, EAST, WEST) must be set"
                )
        except AttributeError:
            if settings.NET_FILE is None:
                raise MintEDGESettingsError(
                    "Either NET_FILE or all bounds (NORTH, SOUTH, EAST, WEST) must be set"
                )

        # Ensure compulsory settings are set
        if settings.BS_DATARATE is None:
            settings.BS_DATARATE = 100e6
            print(
                f"BS_DATARATE not set. Using default value: {settings.BS_DATARATE} bps \n"
            )

        if settings.W_PER_BIT is None:
            raise MintEDGESettingsError("W_PER_BIT must be set in settings.py")

        if settings.MAX_LINK_CAPACITY is None:
            settings.MAX_LINK_CAPACITY = 1e9
            print(
                f"MAX_LINK_CAPACITY not set. Using default value: {settings.MAX_LINK_CAPACITY} bps \n"
            )
        # Currently, servers cannot be manually placed
        if settings.SHARE_OF_SERVERS is None:
            settings.SHARE_OF_SERVERS = 0.5
            print(
                f"SHARE_OF_SERVERS not set. Using default value: {settings.SHARE_OF_SERVERS * 100}% of servers \n"
            )

        if settings.SERVERS is [] or settings.SERVERS is None:
            raise MintEDGESettingsError(
                "At least one SERVER type must be set in settings.py"
            )

        if settings.SERVICES is [] or settings.SERVICES is None:
            raise MintEDGESettingsError(
                "At least one SERVICE type must be set in settings.py"
            )
