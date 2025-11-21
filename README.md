# MintEDGE

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/github/license/blasf1/MintEDGE)
![GitHub repo size](https://img.shields.io/github/repo-size/blasf1/MintEDGE)

## What is MintEDGE?

MintEDGE is a flexible edge computing simulation framework that allows the configuration of various aspects of the infrastructure and enables researchers to test novel energy optimization strategies. MintEDGE offers the following features:

- Integrate **SUMO** [3] for realistic mobility scenarios.
- Stay **agnostic to the Radio Access Network**.
- Define your own resource allocation strategy, focused on energy, QoS or both.
- Use real maps for more realistic scenarios.
- Import **real infrastructures**. Data for the **Netherlands** [5] and **Luxembourg** [6] is included.
- Import realistic **mobility traces** such as **TAPASCologne** [7] or **VehiLux** [8], or generate your own with SUMO.
- Plug in **workload predictors** as part of your resource allocation strategy (an ideal predictor is provided by default).
- We use the realistic and lightweight energy model from [LEAF](https://github.com/dos-group/leaf/)'s [4].

<br>

## Orchestrator Operation and System Model (intuition)

The orchestrator controls three core objects:

- **Status vector**  
  Indicates, for each Base Station (BS), whether it hosts an edge server and whether that server is currently active.

- **Assignation matrix**  
  Describes where requests from each BS are processed.  
  Example: if BS *i* receives 4 requests of service *k* and 2 are handled at server *j* and 2 at server *m*, then the matrix entries for (i, k, j) and (i, k, m) are 0.5 each.

- **Allocation matrix**  
  Describes how CPU resources are divided among services on each server.

By changing these three, you implement different orchestration / energy-saving strategies. MintEDGE gives you a controlled environment to test them.

<br>

## Installation

There are two main ways to run MintEDGE:

1. **Linux (native install, recommended)**: you have to use Ubuntu/Debian based distribution.
2. **macOS & Windows (Docker)**: use this one if you don't want to use Linux.

### 1. Native installation in Linux (Recommended)
MintEDGE has been tested with **Python 3.11.5**.

#### 1.1 Clone the repository:

```bash
git clone https://github.com/blasf1/MintEDGE.git
cd MintEDGE
```

#### 1.2 Create a virtual environment (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
You should now see (.venv) in your terminal prompt.

#### 1.3 Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 1.4 Install SUMO and its dependencies
On Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install -y sumo sumo-tools sumo-doc
```
Make sure the `sumo` binary is on your `PATH`:
```bash
which sumo
sumo --version
```
If `sumo --version` prints a version number (e.g. `1.25.0`), you’re good.

### 2. Windows and MacOS installation using Docker

#### 2.1 Clone the repository:

```bash
git clone https://github.com/blasf1/MintEDGE.git
cd MintEDGE
```

#### 2.2 Build the Docker image
A `Dockerfile`is provided in the repository. It:
- Uses a Python base image
- Installs SUMO and its tools inside the container
- Installs MintEDGE's `requirements.txt`

Build the image (name it `mintedge`):
```bash
docker build -t mintedge .
````

<br>

## Usage

You can run the simulator with the following command:

```bash
python MintEDGE.py --simulation-time 3600 --seed 1 --output results.csv
```

Or, if you are using the Docker way:

```bash
docker run --rm -v "$PWD:/app" mintedge \
  python MintEDGE.py --simulation-time 3600 --seed 1 --output results.csv
```
After the run, you should see `results.csv` (or whatever name you chose) appear in your MintEDGE folder on the host.
You can adjust the simulation time, the seed and the output file in the command line. This facilitates launching multiple simulations simultaneously in distributed environments, e.g. a cluster with SLURM scheduler. More advanced settings about your scenario can be adjusted in the `settings.py` file

<br>

## Project Structure

At the top level you will find (simplified):

- `MintEDGE.py`:
Main entry point. Parses command-line arguments and runs a simulation.

- `settings.py`:
Global configuration file for network infrastructure, services, mobility, simulation timing...

- `mintedge/`:
Package with the simulator's logic, including the orchestrator, the energy model, the infrastructure...

- `scenario/`:
Simulated scenario files (infrastructure and mobility data). By default we provide information on the BSs and (some) backhaul links in the Netherlands.

- `requirements.txt`:
Python dependencies

- `Dockerfile`:
Docker definition for containerized runs (Linux SUMO inside).

<br>

## Configuration
MintEDGE is configured through the `settings.py` file at the root of the project. This file is regular Python; when you run `MintEDGE.py`, it imports `settings.py` and uses the values defined there.

With `settings.py` you adjust:

- simulation timing,
- geographical area and scenario,
- base station and backhaul parameters,
- edge server characteristics,
- services and traffic parameters,

by editing `settings.py`.

Below is a description of each group of parameters.

---

### 1. Simulation parameters

```python
USE_PREDICTOR = False
MEASUREMENT_INTERVAL = 1  # seconds
ORCHESTRATOR_INTERVAL = 60  # how often the orchestrator updates the allocation
CAPACITY_BUFFER = 0.2  # [0, 1] share of extra capacity to allocate
REACTION_THRESHOLD = 0.1  # share of reqs rejected to trigger a new allocation
REACTIVE_ALLOCATION = False  # whether to allocate resources reactively
```

- `USE_PREDICTOR` (boolean):
    - `False`: the orchestrator uses only current/observed load.
    - `True`: the orchestrator additionally uses a workload predictor (by default MintEDGE comes with an ideal predictor) to anticipate future load.

- `MEASUREMENT_INTERVAL` (seconds): How often monitoring information is collected (e.g. load, rejected requests). Smaller values provide finer-grained monitoring at higher computational cost.

- `ORCHESTRATOR_INTERVAL` (seconds): How often the orchestrator recomputes: the status of servers (on/off), the assignation matrix (where to send each request) and the allocation matrix (CPU share per service per server).

- `CAPACITY_BUFFER` (fraction in [0, 1]): Extra capacity reserved on top of the estimated required capacity. Example: if the estimated required capacity is `C`, and `CAPACITY_BUFFER = 0.2`, the orchestrator will aim for `1.2 * C`. This can reduce SLA violations at the cost of higher energy consumption.

- `REACTIVE_ALLOCATION` (boolean):
    - `False`: allocations are updated only every `ORCHESTRATOR_INTERVAL`.
    - `True`: the orchestrator may trigger an additional allocation earlier if too many requests are being rejected.

- `REACTION_THRESHOLD` (fraction in [0, 1]):
Only used when `REACTIVE_ALLOCATION = True`. If the share of rejected requests in a measurement interval exceeds this threshold, the orchestrator runs a new allocation immediately.

### 2. Scenario and geography

```python
PLOT_SCENARIO = False

API_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Enschede + Hengelo's coordinates
NORTH, SOUTH, EAST, WEST = 52.2978, 52.1796, 6.9519, 6.7456

PROVIDER = "vodafone"
BSS_FILE = "./scenario/bss.csv"
LINKS_FILE = "./scenario/links.csv"
```

- `PLOT_SCENARIO`: If `True`, the scenario (the map, BSs...) will be plotted/visualized. Helpful for debugging or understanding the topology.
- `API_MIRRORS`: List of Overpass API endpoints used to query OpenStreetMap when generating scenarios. Multiple mirrors improve robustness when one endpoint is overloaded or you are rate-limited.
- `NORTH, SOUTH, EAST, WEST`: Geographic bounding box for the scenario. Several commented examples are provided (Twente, Enschede, Elburg, Luxembourg, Maastricht). The area within the lines is converted by MintEDGE into a SUMO network file. You can comment this out and provide your own in `NET_FILE`. 
- `PROVIDER`: Name of the mobile network operator for which infrastructure data is modelled (e.g. `"vodafone"`). This can be used to filter infrastructure data in `bss.csv` / `links.csv`.
- `BSS_FILE`: Path to a CSV file with base station information (coordinates, operator, IDs, etc.).
- `LINKS_FILE`: Path to a CSV file with backhaul link information (connections between BSs).

### 3. Mobility and user generation
```python
RANDOM_ROUTES = True
NET_FILE = "./scenario/Luxembourg.net.xml"
ROUTES_FILE = "./scenario/Luxembourg.rou.xml"

NUMBER_OF_CARS = 2500
NUMBER_OF_PEOPLE = 500
NUMBER_OF_STATIONARY = 100

USER_COUNT_DISTRIBUTION = [
    0.13, 0.1, 0.07, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04,
    0.06, 0.08, 0.09, 0.1, 0.1, 0.11, 0.12, 0.12, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.16, 0.15, 0.13
]
```

- `RANDOM_ROUTES`:
    - `True`: routes are generated randomly, i.e., the users will move randomly across the map.
    - `False`: routes from `ROUTES_FILE` are used.

- `NET_FILE`: Path to a SUMO `.net.xml`file representing the road network. Required if no coordinates are provided or if `RANDOM_ROUTES = False`.

- `ROUTES_FILE`: Path to a SUMO `.rou.xml` file with precomputed routes. Required if `RANDOM_ROUTES = False`.
- `NUMBER_OF_CARS`, `NUMBER_OF_PEOPLE`, `NUMBER_OF_STATIONARY`: Total numbers of vehicles, pedestrians and stationary users (like cameras or sensors) that will be generated when `RANDOM_ROUTES = True`.
- `USER_COUNT_DISTRIBUTION`: List with 24+ values representing the relative share of active users per hour of the day. The simulator uses this distribution to vary the number of active users over time, combining it with the totals above. You can adjust this distribution to model different daily patterns (e.g. more traffic in the evening, flatter load, etc.).

### 4. Base Stations

```python
BS_BANDWIDTH = 100e6  # 100 MHz
SNR0_DB = 55          # reference SNR at 1 meter (dB)
SNR0_LIN = 10 ** (SNR0_DB / 10)  # linear SNR at 1 meter
PATHLOSS_EXPONENT = 2.0          # free-space path loss
MIN_USER_RATE = 1e6  # 1 Mbps

BS_DATARATE = BS_BANDWIDTH * np.log2(1.0 + SNR0_LIN)
```
These parameters control how much radio capacity is available at a base station and how it degrades with distance.

- `BS_BANDWIDTH`: Total bandwidth per BS (Hz)
- `SNR0_DB` and `SNR0_LIN`: Reference signal-to-noise ratio at 1 meter in dB and in linear scale.
- `PATHLOSS_EXPONENT`: Path loss exponent in the propagation model.
- `MIN_USER_RATE`: Minimum acceptable user data rate (bps).
- `BS_DATARATE`: Baseline data rate used in preallocations, computed from the Shannon–Hartley theorem. You generally don’t change this directly; it depends on `BS_BANDWIDTH` and `SNR0_LIN`.

### 5. Backhaul network
```python
W_PER_BIT = 5.9e-9       # 5.9 nJ/bit
MAX_LINK_CAPACITY = 10e9  # 10 Gbps
```
- `W_PER_BIT`: Energy consumed by the backhaul per transmitted bit (J/bit). This is used to account for energy in the transport network.
- `MAX_LINK_CAPACITY`: Maximum capacity of a backhaul link (bits/s). This constrains how much traffic can be carried between base stations and edge servers / aggregation points.

### 6. Edge servers
```python
SHARE_OF_SERVERS = 0.5

SERVERS = [
    {
        "MAX_POWER": 696,
        "IDLE_POWER": 222,
        "MAX_CAPACITY": 11260532,
        "BOOT_TIME": 20,
    },
]
```
- `SHARE_OF_SERVERS` (fraction in [0, 1]). Fraction of base stations that host an edge server. Example: with 100 base stations and `SHARE_OF_SERVERS = 0.5`, 50 of them will have an edge server attached.
- `SERVERS`: List of server “types”. Each dictionary defines one server configuration:
    - `"MAX_POWER"`: maximum power consumption of the server (Watts) when running at full capacity.
    - `"IDLE_POWER"`: power consumption when the server is powered on but idle.
    - `"MAX_CAPACITY"`: maximum processing capacity (operations per second, or an equivalent CPU capacity unit).
    - `"BOOT_TIME"`: time required to boot the server (seconds) when turning it on.

    You can define multiple server types (see the commented examples in settings.py). MintEDGE can then use these types to populate the infrastructure.

### 7. Services and service mapping

```python
CAR_SERVICES = ["connected_vehicles"]
PEDESTRIAN_SERVICES = ["augmented_reality", "virtual_reality"]
STATIONARY_SERVICES = ["video_analysis"]

SERVICES = [
    # name, workload(ops/s), lambda(req/s), vin(bytes), vout(bytes), delay_budget(seconds)
    Service("connected_vehicles", 14000, 10, 1600, 100, 5e-3),
    Service("augmented_reality", 50000, 0.5, 1500 * 1024, 25 * 1024, 15e-3),
    Service("video_analysis", 30000, 6, 1500 * 1024, 20, 30e-3),
]
```
There are two levels here:

#### 7.1 Mapping user types to services

- `CAR_SERVICES`:
List of service names that vehicular users can request.

- `PEDESTRIAN_SERVICES`:
Services used by pedestrian users.

- `STATIONARY_SERVICES`:
Services used by stationary users (e.g. cameras, sensors).

These lists contain names that must match the service names defined in the `SERVICES` list below.

#### 7.2 Service definitions

Each Service(...) has the form:
```python
Service(
    name: str,
    workload: float,      # operations per second required per active request
    arrival_rate: float,  # λ, requests per second (average)
    vin: int,             # input size in bytes
    vout: int,            # output size in bytes
    delay_budget: float,  # maximum tolerable latency in seconds
)
```

For example:

`Service("connected_vehicles", 14000, 10, 1600, 100, 5e-3)`

means:

- Name: `"connected_vehicles"`

- CPU demand: 14,000 operations per second per active request

- Request arrival rate: 10 req/s on average

- Input size: 1600 bytes (payload size from user to edge)

- Output size: 100 bytes (response payload)

- Delay budget: 0.005 seconds (5 ms)

To add a new service, you would:

1. Add a new Service(...) entry to SERVICES, e.g.:

`Service("my_new_service", 40000, 2, 2_000, 500, 0.020),`


2. Add "my_new_service" to one (or more) of: `CAR_SERVICES`, `PEDESTRIAN_SERVICES`, `STATIONARY_SERVICES` depending on which user types can request it.

<br>


## Writing your own resource allocation strategy

MintEDGE is designed so that the **resource allocation strategy is a pluggable module**.

By default, the strategy is implemented in `mintedge/allocation_strategy.py`. This is the file you customize

At each orchestration step (every `ORCHESTRATOR_INTERVAL` seconds, or earlier if reactive allocation is enabled), the orchestrator:

1. Collects telemetry (load, rejected requests, delays, utilizations, energy, …).
2. Calls the strategy in `mintedge/allocation_strategy.py`.
3. Receives three outputs:
   - **status vector**: which servers are on/off and where they are deployed,
   - **assignation matrix**: where each BS’s requests are processed,
   - **allocation matrix**: share of CPU capacity each service receives on each server.

By changing how these three objects are computed in `allocation_strategy.py`, you define a new resource allocation / energy-saving strategy.

Inside you will find the default strategy implementation. This file contains the function (or class) that the orchestrator calls every `ORCHESTRATOR_INTERVAL`. Read the docstring and comments in this file first; they describe exactly what inputs you receive and what you must return.


<br>

## How to contribute

Contributions are welcome.

- Keep pull requests **small and focused** (one feature or bug fix per PR).
- Make sure a **basic run** still works (native or Docker), e.g. a short simulation with default settings.
- If you change behaviour, **update documentation** (`README.md`, `settings.py` comments, etc.) accordingly.
- Prefer **clear names, docstrings, and comments** over clever but hard-to-read code.
- In the PR description, briefly state:
  - what you changed,
  - why you changed it,
  - how you verified it (commands / scenarios).

That’s it. Standard GitHub fork/branch/PR workflow applies.

## References

[1] Blas Gómez, Suzan Bayhan, Estefanía Coronado, José Villalón, Antonio Garrido, "MintEDGE: Multi-tier sImulator for eNergy-aware sTrategies in Edge Computing", In Proc. of ACM MobiCom '23, October, 2023. Madrid, Spain. DOI: [10.1145/3570361.3615727](https://dl.acm.org/doi/abs/10.1145/3570361.3615727).

[2] Blas Gómez, Estefanía Coronado, José Villalón, Antonio Garrido, "Energy-focused simulation of edge computing architectures in 5G networks", The Journal of Supercomputing, Vol. 80, pp. 3109-3125, Feb. 2024. DOI: [10.1007/s11227-024-05926-z](https://link.springer.com/article/10.1007/s11227-024-05926-z).

[3] SUMO - Simulation of Urban MObility, [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/). Accessed: 24/07/2023.

[4] Philipp Wiesner and Lauritz Thamsen. "[LEAF: Simulating Large Energy-Aware Fog Computing Environments](https://ieeexplore.ieee.org/document/9458907)" In Proc. of the 2021 5th IEEE International Conference on Fog and Edge Computing (ICFEC). 2021 [[arXiv preprint]](https://arxiv.org/pdf/2103.01170.pdf) [[video]](https://youtu.be/G70hudAhd5M)

[5] Antennekaart, [https://antennekaart.nl](https://antennekaart.nl). Accessed: 15/04/2023.

[6] Etablissements classés - Cadastre GSM. (2023). [Data set]. Administration de l’environnement. [http://data.europa.eu/88u/dataset/etablissements-classes-cadastre-gsm](http://data.europa.eu/88u/dataset/etablissements-classes-cadastre-gsm)

[7] [TAPASCologne project](http://kolntrace.project.citi-lab.fr/). Accessed: 24/07/2023.

[8] Yoann Pigné, Grégoire Danoy, Pascal Bouvry. A Vehicular Mobility Model based on Real Traffic Counting Data. In Thomas Strang et al., editors, Nets4Cars/Nets4Trains 2011, Volume 6596, Series Lecture Notes in Computer Science, Pages 131-142. ISBN: 978-3-642-19785-7. Springer, 2011. [VehiLux](https://vehilux.gforge.uni.lu/)