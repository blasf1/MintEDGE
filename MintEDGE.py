import argparse
import sys

import mintedge

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--simulation-time", required=True, help="time to be simulated"
    )
    parser.add_argument(
        "--seed", required=True, help="seed for the random generator"
    )
    parser.add_argument("--output", required=True, help="output file path")

    args = sys.argv[1:]
    args = parser.parse_args(args)

    sim_time = int(args.simulation_time)
    output = args.output
    seed = int(args.seed)

    sim = mintedge.Simulation(sim_time, output, seed)
    sim.run()
