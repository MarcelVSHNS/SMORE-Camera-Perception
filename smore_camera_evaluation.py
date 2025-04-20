import open3d as o3d
import os
import copy
import numpy as np
import coopscenes as cs
import stixel as stx
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R

from models import StixelPredictor


def to_pointcloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    if isinstance(points, cs.Points):
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
    else:
        xyz = points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def remove_ground(pcd, distance_threshold=0.24, ransac_n=3, num_iterations=1000):
    # Fit plane using RANSAC
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    # Extract inliers and outliers
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)

    return non_ground, ground, plane_model


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def main():
    DATAPATH = "data/seq_2"
    WEIGHTS_PATH_VEH = "models/StixelNExT-Pro_fresh-durian-552_20.pth"
    WEIGHTS_PATH_TOW = "models/StixelNExT-Pro_crisp-eon-553_10.pth"

    # Models
    vehicle_stixel_predictor = StixelPredictor(weights=WEIGHTS_PATH_VEH)
    tower_stixel_predictor = StixelPredictor(weights=WEIGHTS_PATH_TOW)

    # Dataset
    dataset = cs.Dataloader(DATAPATH)
    """
    frames = []
    for datarecord in dataset:
        for frame in datarecord:
            frames.append(frame)
    """
    record = cs.DataRecord(os.path.join(DATAPATH, "id00747_2024-10-05_11-54-47.4mse"))
    frame = record[20]
    # frame.tower.cameras.VIEW_1.show()
    # frame.tower.cameras.VIEW_2.show()
    # frame.vehicle.cameras.STEREO_LEFT.show()

    # Ground Truth
    t_cam_vehicle = frame.vehicle.cameras.STEREO_LEFT.info.extrinsic
    t_vehicle_infra = frame.vehicle.info.extrinsic
    t_infracam_infra = frame.tower.cameras.VIEW_1.info.extrinsic
    ground_truth_t = np.linalg.inv(t_infracam_infra) @ t_vehicle_infra @ t_cam_vehicle  # t_cam_infracam

    # frame.tower.cameras.VIEW_2.image.show()
    # frame.vehicle.cameras.STEREO_LEFT.image.show()

    # """
    # StixelWorld from Vehicle
    stixel_veh = vehicle_stixel_predictor.inference(image=frame.vehicle.cameras.STEREO_LEFT.image.image,
                                                    name=f"{frame.frame_id}_Vehicle",
                                                    camera_info=frame.vehicle.cameras.STEREO_LEFT.info)
    # stx_veh_img = stx.draw_stixels_on_image(stixel_veh)
    # stx_veh_img.show()

    # StixelWorld from Tower
    stixel_tow = tower_stixel_predictor.inference(image=frame.tower.cameras.VIEW_1.image.image,
                                                  name=f"{frame.frame_id}_Tower",
                                                  camera_info=frame.tower.cameras.VIEW_1.info)
    # stx_tow_img = stx.draw_stixels_on_image(stixel_tow)
    # stx_tow_img.show()
    # stx.draw_stixels_in_3d(stixel_tow)

    stx_tow_pts = stx.convert_to_point_cloud(stixel_tow)
    stx_veh_pts = stx.convert_to_point_cloud(stixel_veh)
    pcd_infra = to_pointcloud(stx_tow_pts)
    pcd_vehicle = to_pointcloud(stx_veh_pts)
    # """

    voxel_size = 1
    # === 7. Infrastruktur-Rotation per ICP schätzen
    # pcd_vehicle = to_pointcloud(frame.vehicle.lidars.TOP.points)
    # pcd_infra = to_pointcloud(frame.tower.lidars.UPPER_PLATFORM.points)

    # pcd_vehicle = to_pointcloud(frame.vehicle.lidars.RIGHT.points)
    # pcd_vehicle.transform(frame.vehicle.lidars.RIGHT.info.extrinsic)
    # pcd_infra = to_pointcloud(frame.tower.lidars.UPPER_PLATFORM.points)
    # pcd_infra.transform(frame.tower.lidars.UPPER_PLATFORM.info.extrinsic)

    pcd_vehicle_down, fpfh_vehicle = preprocess_point_cloud(copy.deepcopy(pcd_vehicle), voxel_size)
    pcd_infra_down, fpfh_infra = preprocess_point_cloud(copy.deepcopy(pcd_infra), voxel_size)

    pcd_vehicle.paint_uniform_color([0, 1, 0])  # vehicle = gruen
    pcd_infra.paint_uniform_color([1, 0, 0])    # infra = rot
    o3d.visualization.draw_geometries([pcd_vehicle, pcd_infra])

    # === RANSAC-Matching ===
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd_vehicle_down,
        target=pcd_infra_down,
        source_feature=fpfh_vehicle,
        target_feature=fpfh_infra,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2.0)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    vehicle_tower_tf = cs.Transformation('lidar_top', 'lidar_upper_platform', ransac_result.transformation)

    # === ICP-Verfeinerung auf Basis von RANSAC-Ergebnis ===
    result_icp = o3d.pipelines.registration.registration_icp(
        source=pcd_vehicle,
        target=pcd_infra,
        max_correspondence_distance=voxel_size,
        init=vehicle_tower_tf.mtx,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    icp_tf = cs.Transformation('lidar_top', 'lidar_upper_platform', result_icp.transformation)

    """
    reg_result = o3d.pipelines.registration.registration_icp(
        pcd_infra, pcd_vehicle,
        max_correspondence_distance=5.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    """

    # === 8. ICP-Rotation in T_infra einbauen

    # === 10. Ursprung verschieben (optional)
    pcd_vehicle.transform(icp_tf.mtx)
    pred_mtx = icp_tf.mtx
    gt_mtx = t_vehicle_infra
    error_mtx = np.linalg.inv(gt_mtx) @ pred_mtx

    print(f"Transformation Matrix: \n{pred_mtx}")
    print(f"GT Matrix: \n{gt_mtx}")

    # Translationsfehler berechnen
    translation_error = error_mtx[:3, 3]
    translation_distance = np.linalg.norm(translation_error)

    # Rotationsfehler berechnen (in Grad)
    rotation_matrix = error_mtx[:3, :3]
    rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
    rotation_angle_deg = np.linalg.norm(rotation_vector) * 180 / np.pi

    print("Error matrix (GT⁻¹ * Pred):")
    print(error_mtx)
    print("\nTranslation error (in meter):", translation_distance)
    print("Rotation error (in degree):", rotation_angle_deg)

    # === 11. Einfärben & anzeigen
    o3d.visualization.draw_geometries([pcd_vehicle, pcd_infra])


if __name__ == "__main__":
    main()


"""
# StixelWorld from Vehicle
stixel_veh = vehicle_stixel_predictor.inference(image=frame.vehicle.cameras.STEREO_LEFT.image.image,
                                                name=f"{frame.frame_id}_Vehicle",
                                                camera_info=frame.vehicle.cameras.STEREO_LEFT.info)
# stx_veh_img = stx.draw_stixels_on_image(stixel_veh)
# stx_veh_img.show()

# StixelWorld from Tower
stixel_tow = tower_stixel_predictor.inference(image=frame.tower.cameras.VIEW_1.image.image,
                                              name=f"{frame.frame_id}_Tower",
                                              camera_info=frame.tower.cameras.VIEW_1.info)
# stx_tow_img = stx.draw_stixels_on_image(stixel_tow)
# stx_tow_img.show()
# stx.draw_stixels_in_3d(stixel_tow)

transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
x1, y1 = transformer.transform(float(9.2971638),
                               float(48.7407864))
x2, y2 = transformer.transform(float(9.2972455),
                               float(48.7407824))
GPS_vehicle = np.array([x2, y2, frame.vehicle.info.height[2, 3]])
GPS_infra = np.array([x1, y1, frame.tower.info.height[2, 3]])
print(f"GPS Vehicle: {GPS_vehicle}")
print(f"GPS Infra: {GPS_infra}")

stx_tow_pts = stx.convert_to_point_cloud(stixel_tow)
stx_veh_pts = stx.convert_to_point_cloud(stixel_veh)
#pcd1 = to_pointcloud(stx_tow_pts)
#pcd2 = to_pointcloud(stx_veh_pts)
pcd1 = to_pointcloud(frame.tower.lidars.UPPER_PLATFORM.points)
pcd2 = to_pointcloud(frame.vehicle.lidars.TOP.points)
translation = np.array([x2 - x1, y2 - y1, frame.vehicle.info.height[2, 3] - frame.tower.info.height[2, 3]])
#pcd2.translate(translation)

# o3d.visualization.draw_geometries([pcd1.paint_uniform_color([1, 0, 0]), pcd2.paint_uniform_color([0, 1, 0])])
pcd1, ground1, model1 = remove_ground(pcd1)
pcd2, ground2, model2 = remove_ground(pcd2)
o3d.visualization.draw_geometries([pcd1.paint_uniform_color([1, 0, 0]), pcd2.paint_uniform_color([0, 1, 0])])

reg_result = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1,
    max_correspondence_distance=200.0,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Schritt 1: GPS-basierte Translation
t_vehicle_infra_translation = np.eye(4)
t_vehicle_infra_translation[:3, 3] = np.array(GPS_vehicle) - np.array(GPS_infra)
print("Translation Matrix:")
print(t_vehicle_infra_translation)

# Schritt 2: Rotation und Feinkorrektur aus ICP
t_vehicle_infra_est = reg_result.transformation @ t_vehicle_infra_translation

# Endgültige geschätzte Transformation Kamera → Infrastrukturkamera:
t_cam_infracam_est = np.linalg.inv(t_infracam_infra) @ t_vehicle_infra_est @ t_cam_vehicle

print("Transformation Matrix:")
print(t_vehicle_infra_est)
print("GT Matrix:")
print(t_vehicle_infra)
print("Error Matrix:")
print(np.linalg.inv(ground_truth_t) @ t_cam_infracam_est)

# Optional: Transformation anwenden
pcd2.transform(t_vehicle_infra_est)
o3d.visualization.draw_geometries([pcd1.paint_uniform_color([1, 0, 0]), pcd2.paint_uniform_color([0, 1, 0])])
"""
