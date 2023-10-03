"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Matterport Semantics Callback
"""

# python
import asyncio

# omni
import carb

# omni-isaac-core
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.matterport.config import MatterportConfig

# omni-isaac-matterport
from .matterport_sem import MatterportWarp


class MatterportCallbackDomains:
    """
    Callback to render semantics
    """

    def __init__(
        self,
        cfg: MatterportConfig,
        domains: MatterportWarp,
    ) -> None:
        """Render Callback for Matterport Semantics"""
        # config
        self._cfg: MatterportConfig = cfg

        # inti semantics
        self.domains: MatterportWarp = domains

        # interal parameters
        self._save_counter: int = 0
        if SimulationContext.instance():
            self._sim: SimulationContext = SimulationContext.instance()
        else:
            carb.log_error("No Simulation Context found! Matterport Callback not attached!")

        # remove Isaac Semantic rendering for the selected cameras
        # NOTE: necessary, otherwise matterport data will be overwritten when updated
        for cam in self.domains.cameras:
            del cam.omni_camera._rep_registry["semantic_segmentation"]
        return

    ##
    # Callback Setup
    ##

    def set_domain_callback(self, val) -> None:
        if val:
            self._sim.pause()
            self._sim.add_physics_callback("mp_sem_callback", callback_fn=self._compute_domains)
        else:
            if self._cfg.save:
                self.domains._end_save()
            self._sim.add_physics_callback("mp_sem_callback")
        return

    ##
    # Callback Function
    ##

    def _compute_domains(self, dt: float) -> None:
        if int(self._save_counter % self._cfg.compute_frequency) == 0:
            # reset counter to prevent overflow
            self._save_counter = 0
            # get semantics
            time_step = int(self._sim.current_time / self._sim.get_physics_dt())
            asyncio.ensure_future(self.domains.compute_domains(time_step))
        self._save_counter += 1
        return


# EoF
