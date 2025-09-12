# MintEDGE

![Python](https://img.shields.io/badge/Python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![GitHub](https://img.shields.io/github/license/blasf1/MintEDGE)
![GitHub repo size](https://img.shields.io/github/repo-size/blasf1/MintEDGE)

## What is MintEDGE?

MintEDGE is a flexible edge computing simulation framework that allows the configuration of various aspects of the infrastructure and enables researchers to test novel energy optimization strategies. MintEDGE offers the following features:

- SUMO [3] integration for realistic mobility scenarios.
- Agnostic w.r.t. Radio Access Network
- Define your own resource allocation strategy, focused on energy, QoS or both.
- Use real maps for more realistic scenarios.
- Ability to import real infrastructures. We provide data for the Netherlands [5] and Luxembourg [6].
- Import realistic mobility traces such as TAPASCologne [7] or VehiLux [8], or generate yours with SUMO or any other user mobility simulator.
- You can use MintEDGE to evaluate workload predictors as part of your resource allocation strategy. By default, we provide an ideal predictor.
- We use the realistic and lightweight energy model from [LEAF](https://github.com/dos-group/leaf/)'s [4].

<br>

## Orchestrator Operation and System Model

The operation of the orchestrator is based on the control of 2 matrixes and a vector:

- Status vector: The status vector indicates whether a Base Station (BS) hosts an edge server or not and whether the server is active.
- Assignation matrix: The assignation matrix represents where the requests received at each BS are attended, i.e., to which server are assigned. For instance, if a BS _i_ receives 4 requests for service _k_, and then 2 of them are processed at server _j_ and other 2 are processed at server _m_, then the value of the components _i,k,j_ and _i,k,m_ of the assignation matrix will be 0.5 each (half of the requests are processed in each destination server).
- Allocation matrix: Represents the share of CPU operations assigned to each service.

Tuning these three parameters, the orchestrator controls the operation of the edge infrastructure. MintEDGE makes it easy to find new energy efficient strategies that also take QoS into account.

<br>

## Installation

MintEDGE has been tested using Python 3.11.5. We recommend this version.

1. Clone the repository

```bash
git clone https://github.com/blasf1/MintEDGE.git
```

2. Go to the downloaded directory and install the requirements

```bash
cd MintEDGE
pip install -r requirements.txt
```

3. Install SUMO and its dependencies:

```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

You can install it in your conda environment with the following command:

```bash
conda install -c blasf1 sumo
```

<br>

## Usage

You can run the simulator with the following command:

```bash
python MintEDGE.py --simulation-time 3600 --seed 1 --output results.parquet
```

You can adjust the simulation time, the seed and the output file in the command line. This facilitates launching multiple simulations simultaneously in distributed environments, e.g. a cluster with SLURM scheduler. Other settings can be adjusted in the settings.py file

Test your own efficient resource allocation or energy efficiency strategy by replacing the allocation_strategy.py file in the mintedge directory with your own.

<br>

## References

[1] Blas Gómez, Suzan Bayhan, Estefanía Coronado, José Villalón, Antonio Garrido, "MintEDGE: Multi-tier sImulator for eNergy-aware sTrategies in Edge Computing", In Proc. of ACM MobiCom '23, October, 2023. Madrid, Spain. DOI: [10.1145/3570361.3615727](https://dl.acm.org/doi/abs/10.1145/3570361.3615727).

[2] Blas Gómez, Estefanía Coronado, José Villalón, Antonio Garrido, "Energy-focused simulation of edge computing architectures in 5G networks", The Journal of Supercomputing, Vol. 80, pp. 3109-3125, Feb. 2024. DOI: [10.1007/s11227-024-05926-z](https://link.springer.com/article/10.1007/s11227-024-05926-z).

[3] SUMO - Simulation of Urban MObility, [https://www.eclipse.org/sumo/](https://www.eclipse.org/sumo/). Accessed: 24/07/2023.

[4] Philipp Wiesner and Lauritz Thamsen. "[LEAF: Simulating Large Energy-Aware Fog Computing Environments](https://ieeexplore.ieee.org/document/9458907)" In the Proceedings of the 2021 5th IEEE International Conference on Fog and Edge Computing (ICFEC). 2021 [[arXiv preprint]](https://arxiv.org/pdf/2103.01170.pdf) [[video]](https://youtu.be/G70hudAhd5M)

[5] Antennekaart, [https://antennekaart.nl](https://antennekaart.nl). Accessed: 15/04/2023.

[6] Etablissements classés - Cadastre GSM. (2023). [Data set]. Administration de l’environnement. [http://data.europa.eu/88u/dataset/etablissements-classes-cadastre-gsm](http://data.europa.eu/88u/dataset/etablissements-classes-cadastre-gsm)

[7] [TAPASCologne project](http://kolntrace.project.citi-lab.fr/). Accessed: 24/07/2023.

[8] Yoann Pigné, Grégoire Danoy, Pascal Bouvry. A Vehicular Mobility Model based on Real Traffic Counting Data. In Thomas Strang et al., editors, Nets4Cars/Nets4Trains 2011, Volume 6596, Series Lecture Notes in Computer Science, Pages 131-142. ISBN: 978-3-642-19785-7. Springer, 2011. [VehiLux](https://vehilux.gforge.uni.lu/)
