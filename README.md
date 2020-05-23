# waypoints_planning_admm
A deep Q-network to plan waypoints for UAVs avoiding obstacles in indoor environment. 
Applied Alternating Direction Method of Multipliers (ADMM) based weight pruning for the C3D network.

The environment state can be encoded into two different structures: 1D vector or 3D block.
* The 1D vector is composed of normalized distance difference of each obstacle to the destination point with the OBSTACLE_REWARD, followed by distance difference between current position with destination point with GOAL_REWARD. 

* The entire 3D environment is divided into N × N × N grids. The environment is described by a function M() maps a grid (x, y, z) to a real value M : (x, y, z) → R. A grid, g, that contains obstacle will be mapped to -0.1, M(g) = −0.1. The destination grid that the agent needs to reach is mapped to 1, and the grid where the UAV is currently located is mapped to 0.1. All other grids are mapped to zeros in the discretized environment block.

A UAV can choose any of the 3 × 3 × 3 grids around its current location as the next waypoint.

### Prerequisites

Packages you need to install the software and how to install them

```
numpy 1.16.4
Pytorch 1.1.0
torchvision
pillow==6.2.1
pyyaml
python-lmdb
pyarrow==0.11.1
tqdm
```

## Training and Evaluation
See the [command.md](command.md) file for details

## Weights
Saved weights in ```./saved_weights/ ``` under current directory

## License
See the [License.md](License.md) file for details