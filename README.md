<div style="display: flex;">
    <img src="docs/example_matterport.png" alt="Matterport Mesh" style="width: 48%; padding: 5px;">
    <img src="docs/example_carla.png" alt="Unreal Engine / Carla Mesh" style="width: 48%; padding: 5px;">
</div>

---

# Omniverse Semantic Enviornment Extension

[![IsaacSim](https://img.shields.io/badge/IsaacSim-2023.1.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Orbit](https://img.shields.io/badge/Orbit-0.3.0-silver)](https://isaac-orbit.github.io/orbit/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

Extension for omniverse isaac sim that allows to simply load environments from e.g. Matterport3D or Unreal Engine, and either use their semantic information or quickly augment them. 
In addition, it includes tasks to quickly render images from different domains with different field of views or sample trajectories.
It can be used either with standalone scripts or in an extension workflow using the GUI. 
They are developed as part of the ViPlanner project ([Paper](https://arxiv.org/abs/2310.00982) | [Code](https://github.com/leggedrobotics/viplanner))
and are based on the [Orbit](https://isaac-orbit.github.io/) framework.

## Installation

To install the extensions, follow these steps:

1. Install Isaac Sim using the [Orbit installation guide](https://isaac-orbit.github.io/orbit/source/setup/installation.html).
2. Clone the orbit repo and link the extension.

```
git clone git@github.com:NVIDIA-Omniverse/orbit.git
cd orbit/source/extensions
ln -s {ORBIT_ENVS_PROJECT_DIR}/extensions/omni.isaac.orbit_envs .
ln -s {ORBIT_ENVS_PROJECT_DIR}/extensions/omni.isaac.carla .
```

4. Then run the orbit installer script.

```
./orbit.sh -i
```

Always use the matterport raycast camera. Matterport meshes are loaded as many different meshes which is currentlt not supported in Orbit.
Instead, the MatterportRaycaster uses the ply mesh which is a single mesh.

## Usage


### Extension Workflow

For the Matterport extension, a GUI interface is available. To use it, start the simulation:

```
./orbit.sh -s
```

Then, in the GUI, go to `Window -> Extensions` and type `Matterport` in the search bar. You should see the Matterport3D extension.
Enable it to open the GUI interface.

To use both as part of an Orbit workflow, please refer to the [ViPlanner Demo](https://github.com/leggedrobotics/viplanner/tree/main/omniverse).





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
