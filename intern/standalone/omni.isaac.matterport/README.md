# Omniverse Matterport Extension



### Known issues

- `RuntimeError: Invalid device string: 'cuda:-1'` when used in extension mode. Fix by running the simulator with the
  additional argument `--/physics/cudaDevice=0`
