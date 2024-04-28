# Omniverse Matterport Extension



Always use the matterport raycast camera. Matterport meshes are loaded as many different meshes which is currentlt not supported in Orbit. 
Instead, the MatterportRaycaster uses the ply mesh which is a single mesh. 

### Known issues

- `RuntimeError: Invalid device string: 'cuda:-1'` when used in extension mode. Fix by running the simulator with the
  additional argument `--/physics/cudaDevice=0`
