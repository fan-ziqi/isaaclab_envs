"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

import matplotlib.pyplot as plt

# python
import numpy as np

# omni
import omni
import omni.replicator.core as rep
import scipy.spatial.transform as tf
import torch
from omni.isaac.core.simulation_context import SimulationContext

# omni-isaac-matterport
from omni.isaac.matterport.semantics import MatterportWarp


async def test_depth_warp(domains: MatterportWarp):
    assert len(domains.cameras) == 1, "Only one camera necessary for random exploration"
    cam_data = domains.cameras[0]

    cam_prim = cam_data.prim
    cam_transform = omni.usd.utils.get_world_transform_matrix(cam_prim)
    base_rot = np.array(cam_transform.ExtractRotationMatrix()).T
    cam_pos = torch.tensor(cam_transform.ExtractTranslation(), device="cuda")
    cam_rot = tf.Rotation.from_matrix(base_rot).as_euler(seq="xyz", degrees=True)

    cam_rot_apply = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 180.0]])
    cam_rot_apply_mat = torch.tensor(
        tf.Rotation.from_euler("xyz", cam_rot_apply[2], degrees=True).as_matrix(), device="cuda"
    ).type(torch.float32)

    # get new ray directions in world frame
    cam_data.ray_directions = domains._get_ray_directions(cam_pos, cam_rot_apply_mat, cam_data.pixel_coords)

    # raycast
    cam_data.ray_hit_coords, cam_data.ray_face_indices, cam_data.ray_distances = domains._raycast(
        cam_pos.repeat(len(cam_data.ray_directions)),
        cam_data.ray_directions,
        cam_rot=cam_rot_apply_mat,
        pix_offset=cam_data.pixel_offset,
    )

    # filter inf values
    hit_rate = torch.isfinite(cam_data.ray_distances).sum() / len(cam_data.ray_distances)
    print("Rate of rays hitting the mesh: ", hit_rate)
    cam_data.ray_hit_coords[~torch.isfinite(cam_data.ray_hit_coords)] = 0
    cam_data.ray_distances[~torch.isfinite(cam_data.ray_distances)] = 0

    # get depth_map
    await domains._render_async(cam_data)

    # get replicator image
    rep_depth = rep.create.render_product(cam_prim.GetPrimPath(), resolution=(cam_data.width, cam_data.height))
    depth_front = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")  # ("distance_to_camera")
    depth_front.attach(rep_depth)

    sim = SimulationContext.instance()
    sim.play()

    rep_depth_img = depth_front.get_data(device="cpu")
    rep_depth_img = np.expand_dims(rep_depth_img, axis=2)

    sim.pause()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Depth images")
    ax1.imshow(cam_data.render_depth)
    ax2.imshow(rep_depth_img)
    ax3.imshow(np.abs(cam_data.render_depth / 1000) - np.abs(rep_depth_img))

    # compare images
    if cam_data.render_depth == rep_depth:
        print("Depth maps are equal.")
    else:
        print("Depth maps are not equal.")


# EoF
