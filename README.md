<div style="display: flex;">
    <img src="docs/example_matterport.png" alt="Matterport Mesh" style="width: 48%; padding: 5px;">
    <img src="docs/example_carla.png" alt="Unreal Engine / Carla Mesh" style="width: 48%; padding: 5px;">
</div>

---

# Isaac Navigation Suite

[![IsaacSim](https://img.shields.io/badge/IsaacSim-2023.1.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Orbit](https://img.shields.io/badge/Orbit-0.3.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

**Isaac Navigation Suite** is a framework for robotic navigation task. It is meant to unify navigation-relevant environments,
data-sampling approaches to development planner and a suite to test them. Currently, the suite includes three extensions:

- ``orbit.nav.importer``: Load navigation relevant environments from e.g. Matterport3D or Unreal Engine, and access all their domains (semantic, geometric, RGB, ...).
- ``orbit.nav.collectors``: Sample data from the environments (such as trajectories or rendered images at different viewpoints)
- ``orbit.nav.tasks``: Implementation of different planner and evaluation methods for benchmarkting (coming soon)

The current version was developed as part of the ViPlanner project ([Paper](https://arxiv.org/abs/2310.00982) | [Code](https://github.com/leggedrobotics/viplanner))
and are based on the [Orbit](https://isaac-orbit.github.io/) framework.

## Installation

The extensions are available directly under `Window -> Extensions -> orbit.nav.[importer, collectors, tasks]`.
This workflow is recommended if you want to tryout the extension or simply import meshes.
However, for more advanced usage, we recommend to do the following installation steps:

1. Install Isaac Sim using the [Orbit installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).
2. Clone the orbit repo as well as this repo. The extension is developed using ``Orbit v0.3``.

TODO: (@pascal-roth) update the link to the extension and make sure orbit v0.3

```bash
git clone git@github.com:NVIDIA-Omniverse/orbit.git
git clone git@github.com:leggedrobotics/isaac-nav-suite.git
cd <path-to-your-orbit-repo>
git checkout v0.3

```

3. Link the extensions into the orbit extension directory

```bash
cd <path-to-your-orbit-repo>/source/extensions
ln -s <path-to-your-nav-suite-repo>/extensions/orbit.nav.importer .
ln -s <path-to-your-nav-suite-repo>/extensions/orbit.nav.collector .
ln -s <path-to-your-nav-suite-repo>/extensions/orbit.nav.tasks .
```

4. Then run the orbit installer script.

```bash
cd <path-to-your-orbit-repo>
./orbit.sh -i
```

## Usage


### GUI Workflow

For all extensions a GUI is available with access to certain features. To use it, start the simulation:

```bash
cd <path-to-your-orbit-repo>
./orbit.sh -s
```

Then, in the GUI, go to `Window -> Extensions` and type `orbit.nav.[importer, collectors, tasks]` in the search bar.
Toggle the extensions tat you want to activate. To see the GUI, then select the extensions under `Window` tab.
For a detailed insights in the different GUI interfaces please see

- [Importer GUI README](extensions/orbit.nav.importer/docs/README.md)
- [Data Collectors GUI README](extensions/orbit.nav.collectors/docs/README.md)
- [Tasks GUI README](extensions/orbit.nav.tasks/docs/README.md)

Please be aware, that for new environments, data domains, and more, the configs files of the extensions have to changed.
Not all parameters are included in the GUI, so changes in the files will be necessary.

### Standalone Script Workflow

Standalone scripts can be used to custmize the functionalities and easily integrate different parts of the extensions for your own projects.
Here we provide a set of examples that demonstrate how to use the different parts:

- ``orbit.nav.importer``
  - [Import a Matterport3D Environment](standalone/orbit.nav.importer/check_matterport_import.py)
  - [Import a Carla (Unreal Engine) Environment](standalone/orbit.nav.importer/check_carla_import.py)
  - [Import the Nvidia Warehouse Environment](standalone/orbit.nav.importer/check_warehouse_import.py)
- ``orbit.nav.collectors``
  - [Sample Trajectories from Matterport](standalone/orbit.nav.collectors/check_matterport_trajectory_sampling.py)
  - [Sample Viewpoints and Render Images from Carla (Unreal Engine)](standalone/orbit.nav.collectors/check_carla_viewpoint_sampling.py)
- ``orbit.nav.tasks``
  - available soon


## <a name="CitingViPlanner"></a>Citing

If you use this code in a scientific publication, please cite the following [paper](https://arxiv.org/abs/2310.00982):
```
@article{roth2023viplanner,
  title     ={ViPlanner: Visual Semantic Imperative Learning for Local Navigation},
  author    ={Pascal Roth and Julian Nubert and Fan Yang and Mayank Mittal and Marco Hutter},
  journal   = {2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  month     = {May},
}
```

### License

This code belongs to the Robotic Systems Lab, ETH Zurich.
All right reserved

**Authors: [Pascal Roth](https://github.com/pascal-roth)<br />
Maintainer: Pascal Roth, rothpa@ethz.ch**

This repository contains research code, except that it changes often, and any fitness for a particular purpose is disclaimed.
