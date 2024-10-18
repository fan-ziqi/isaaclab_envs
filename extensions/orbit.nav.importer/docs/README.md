# Navigation Suite Environment Importer

This extensions allows to import diverse environments into Isaac Sim and make their domains available for downstream tasks.


## Extension GUI

To activate the GUI start Isaac Sim, e.g. over

```bash
cd <path-to-your-orbit-repo>
./orbit.sh -s
```

Then navigate to `Window -> Extensions` and type `orbit.nav.importer` in the search and toggle the extension. Next activate the GUI under `Window`.

The interface shows at first a multiple options which environment should be imported. The subsequent interface is adjusted based on the used kind:

TODO: (@pascal-roth) include images of the different extensions

<details>
    <summary><strong>``matterport``</strong></summary>

    When importing matterport meshes, you can either input the original ``.obj`` or an already converted ``.usd`` file. In addition, a ``.ply`` file can be defined which is necessary to access the semantic domain and allow for faster access to the geometric domain. In the case, the mesh should only be investigated, no definition is necessary. Following some mesh parameters can be defined which are relevant when employing a robot in the environment.

    **Semantic Domain**: Regarding the semantic domain, Matterport3D comes with a set of very detailed set of classes. Per default, we use the reduced set of 40 classes with their colors defined [here](../data/matterport/mpcat40.tsv). The custom sensors access the information and make them available for further processing.

    Press the ``load`` button to load the environment. The ``reset`` button is available to remove all elements from the scene.

</details>

<details>
    <summary><strong>``multi-mesh-usd``</strong></summary>

    For any ``.usd`` file that consist out of multiple meshes, use this type. First define the file location, then some mesh parameters can be defined which are relevant when employing a robot in the environment.

    **Semantic Domain**: Most environments consist of multiple meshes. The extensions provides an easy name to class mapping tool, whereas all meshes that include defined string will be assigned a certain class. Such a mappai

    Press the ``load`` button to load the environment. The ``reset`` button is available to remove all elements from the scene.

</details>

<details>
    <summary><strong>``single-mesh-usd``</strong></summary>

    Single meshes allow for a speedup when accessing the different domains. Similar to the multi-mesh-usd setup, first, define the file location, then some mesh parameters which are relevant when employing a robot in the environment. Interesting meshes can be generated with a terrain-generator availe under https://github.com:leggedrobotics/terrain-generator

    **Semantic Domain** The semantic domain for such meshes is currently not supported.

    Press the ``load`` button to load the environment. The ``reset`` button is available to remove all elements from the scene.

</details>

<details>
    <summary><strong>``generated``</strong></summary>

    The Orbit Framework allows to generate environments and randomize their parameters. Also these environments can be used. To do so, define the ``module`` (e.g. ``omni.isaac.lab.terrains.config.rough``) and the config ``class`` (e.g. ``ROUGH_TERRAINS_CFG``). In addition, some terrain parameters can be set for deploying a robot on the terrain.

    **Semantic Domain** The semantic domain for such meshes is currently not supported.

    Press the ``load`` button to load the environment. The ``reset`` button is available to remove all elements from the scene.

</details>


## Standalone scripts

Standalone scripts are provided to demonstrate the loading of different environments:

- [Import a Matterport3D Environment](standalone/orbit.nav.importer/check_matterport_import.py)
- [Import a Carla (Unreal Engine) Environment](standalone/orbit.nav.importer/check_carla_import.py)
- [Import the Nvidia Warehouse Environment](standalone/orbit.nav.importer/check_warehouse_import.py)


::attention::

    For ``matterport`` meshes, always use the custom sensors.
    Matterport meshes are loaded as many different meshes which is currentlt not supported in Orbit.
    Instead, the ``MatterportRaycaster`` and ``MatterportRayCasterCamera`` uses the ply mesh which is a single mesh that additionally includes the semantic domain.
