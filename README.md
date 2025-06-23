# master-thesis

This repository contains code for estimating sea ice drift using real-world data and a simple simulator that I developed. The code is structured so that the simulation workflow and the real-data workflow are separate, making it more user-friendly. This allows you to run the appropriate version without manually changing internal parameters.

Each workflow has its own version of the main script, floe logic, and grid setup. Simply run the correct set of files for the use case you're working on.

## Project Structure

- `main.py`: Main script for **real-world data**
- `main_sim.py`: Main script for **simulation**
- `Floe.py`: The relevant classes and methods used in the thesis, for for real data
- `Floe_sim.py`: The relevant classes and methods used in the thesis, for simulation
- `Grid.py`: Grid setup for real data
- `Grid_sim.py`: Grid setup for simulation
- `Simulator.py`: The logic behind the simulator
- `FloePlot.py`: Shared plotting functions for both workflows
- `intrinsics.yaml`: Camera intrinsics used for the real data
- `data set/`: Contains the first 30 timesteps of real-world data
- `satellite_image.tif`: Cropped satellite image used for visualization

## How to Run

To get started, clone the repository and run either `main.py` or `main_sim.py` depending on whether you want to use real-world data or the simulator.

Feel free to play around with the code. There are optional plots and scenarios you can enable by uncommenting lines I've marked in the scripts.

If you have any questions or run into issues, don’t hesitate to reach out at bendiaa@stud.ntnu.no :)
