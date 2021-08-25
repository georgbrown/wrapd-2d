# WRAPD

[WRAPD: Weighted Rotation-aware ADMM for Parameterization and Deformation](https://www-users.cse.umn.edu/~brow2327/wrapd/)  
[George E. Brown](http://www-users.cs.umn.edu/~brow2327/), University of Minnesota  
[Rahul Narain](http://rahul.narain.name/), Indian Institute of Technology Delhi

## Notes

This project contains a submodule:

- [libigl](https://github.com/libigl/libigl)

## Dependencies

**Required:** Git, CMake, BLAS, LAPACK, GFORTRAN, OpenMP

**Strongly recommended:** [PARDISO](https://www.pardiso-project.org/)

On Ubuntu the required depencencies can be installed with the following command:

    sudo apt-get install git cmake libx11-dev mesa-common-dev libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev libxi-dev libxmu-dev libxinerama-dev libxcursor-dev libblas-dev liblapack-dev libgfortran-9-dev

PARDISO is strongly recommended. Without PARIDSO our algorithm will perform worse than we reported in the paper. If PARDISO is not detected then our algorithm falls back to using Eigen for linear solver operations.

**To install PARDISO:**
Go to https://www.pardiso-project.org/ and follow the instructions for downloading the library and configuring your license.
Note that you will need to place a license file in your root user directory.

## Installation

1. Verify that all the dependencies are installed (see above) 
2. Clone the repository and go into to the project root directory (`wrapd-2d`).
3. Run `git submodule update --init`
4. Copy the PARDISO library file (ending in `.so`) to `wrapd-2d/deps/pardiso/`
5. Run `mkdir build && cd build && cmake .. && make -j`
6. Go back to the `wrapd-2d` directory

## Using the software (example)

To run the application you will need to specify the number of OpenMP threads to be used.
This is done by prepending the command with `OMP_NUM_THREADS=N`, where *N* is number of threads provided to the application.
For example, to run the software with 8 threads, type:

    OMP_NUM_THREADS=8 ./build/param -i <input_mesh.obj>

Example input meshes from the paper are located in: `wrapd-2d/samples/data/`

The output mesh and data will be saved into a directory in `wrapd-2d/output/`.

## Optional arguments

There are many optional arguments that can be provided to tune the algorithm and enable/disable particular features. For a complete list of options type:

    ./build/param --help

## Viewer

We have included a LIBIGL viewer to visualize the evolution of 2D embeddings across solver iterations. Triangle distortions in the embedding are color-mapped from grey to red (redder means more distorted).

When using the viewer, the solver may be started/paused by pressing spacebar. It is also possible to advance a single iteration at a time by pressing 'i'.

Note: The viewer is enabled by default. To disable it and run the code headless, pass the argument `-viewer 0`.

## Paper abstract

Local-global solvers such as ADMM for elastic simulation and geometry optimization struggle to resolve large rotations such as bending and twisting modes, and large distortions in the presence of barrier energies. We propose two improvements to address these challenges. First, we introduce a novel local-global splitting based on the polar decomposition that separates the geometric nonlinearity of rotations from the material nonlinearity of the deformation energy. The resulting ADMM-based algorithm is a combination of an L-BFGS solve in the global step and proximal updates of element stretches in the local step. We also introduce a novel method for dynamic reweighting that is used to adjust element weights at runtime for improved convergence. With both improved rotation handling and element weighting, our algorithm is considerably faster than state-of-the-art approaches for quasi-static simulations. It is also much faster at making early progress in parameterization problems, making it valuable as an initializer to jump-start second-order algorithms.
