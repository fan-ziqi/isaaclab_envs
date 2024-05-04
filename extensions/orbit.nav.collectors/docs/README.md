# Navigation Suite Data Collectors

This extensions allows to collect data from the previously loaded environments and all their domains.


## Extension GUI

To activate the GUI start Isaac Sim, e.g. over

```bash
cd <path-to-your-orbit-repo>
./orbit.sh -s
```

Then navigate to `Window -> Extensions` and type `orbit.nav.collectors` in the search and toggle the extension. Next activate the GUI under `Window`.

Before executing any data collection task, an environment has to be imported. Therefore, the first page of the GUI is the same as for the ``orbit.nav.importer`` extensions with detailed description available [here](../../orbit.nav.importer/docs/README.md).

However, in addition, cameras can now be added to the scene. For the cameras, the semantic and geometric domain are available (for access to the RGB domain, please use standalone scripts). Furthermore, the intrinsics of the cameras are defined. While for the


Once an environment is loaded, the GUI will automatically switch and two example tasks are shown:

- ``Trajectory sampling``:
    For trajectory sampling the environment is first analysed and a graph is constructed in the traversable space. What is traversable can be defined in the corresponding [config file](../orbit/nav/collectors/collectors/terrain_analysis_cfg.py).

    **Important** for the analysis also regarding the semantic domain, a semantic class to cost mapping has to be defined in the config. Per default, an example cost map for ``matterport`` environments is selected.

    Then trajectories (i.e. start-goal pairs and their length) are sampled from the graph. You can define the ``module`` and the ``class`` of the parameters config that is used for the sampling. An example is provided that is optimized for the legged robot ANYmal and a matterport environment.

    The trajectory sampling can be executed multiple times with different number of sampled trajectories as well as different minimum and maximum lengths.

- ``Viewpoint sampling and image rendering``
    For the viewpoint sampling the same terrain analysis as for the trajectory sampling is executed. The graph and traversability parameters are defined in corresponding [config file](../orbit/nav/collectors/collectors/terrain_analysis_cfg.py).

    **Important** for the analysis also regarding the semantic domain, a semantic class to cost mapping has to be defined in the config. Per default, an example cost map for ``matterport`` environments is selected.

    Each node of the prah is a possible viewpoint, with the orientation uniformly sampled between variable bounds. The exact parameters of the sampling can be defined [here](../orbit/nav/collectors/collectors/viewpoint_sampling_cfg.py).  You can define the ``module`` and the ``class`` of the parameters config that is used for the sampling. An example is provided that is optimized for the legged robot ANYmal and a matterport environment. Please not that this configuration assumes that two cameras are added where the first one has access to semantic information and the second to geoemtric information.

    The number of viepoints that are sampled can be directory defined in the GUI. With the button ``Viewpoint Samping`` the viewpoints are saved as ``camera_poses`` under the defined directory. Afterwards, click ``Viewpoint Renedering`` to get the final rendered images. The resulting folder structure is as follows:

    ``` graphql
    cfg.data_dir
    ├── camera_poses.txt                    # format: x y z qw qx qy qz
    ├── cfg.depth_cam_name                  # required
    |   ├── intrinsics.txt                  # K-matrix (3x3)
    |   ├── distance_to_image_plane         # annotator
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    ├── cfg.depth_cam_name                  # optional
    |   ├── intrinsics.txt                  # K-matrix (3x3)
    |   ├── distance_to_image_plane         # annotator
    |   |   ├── xxxx.png                    # images saved with 4 digits, e.g. 0000.png
    ```


## Standalone scripts

Standalone scripts are provided to demonstrate the loading of different environments:

  - [Sample Trajectories from Matterport](standalone/orbit.nav.collectors/check_matterport_trajectory_sampling.py)
  - [Sample Viewpoints and Render Images from Carla (Unreal Engine)](standalone/orbit.nav.collectors/check_carla_viewpoint_sampling.py)


::attention::

    For ``matterport`` meshes, always use the custom sensors.
    Matterport meshes are loaded as many different meshes which is currentlt not supported in Orbit.
    Instead, the ``MatterportRaycaster`` and ``MatterportRayCasterCamera`` uses the ply mesh which is a single mesh that additionally includes the semantic domain.
