import open3d as o3d
import os
import copy
import numpy as np
import pandas as pd
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
    for sequence in ["1",]:
        DATAPATH = f"/home/marcel/workspace/datasets/raw/coopscenes/validation/seq_{sequence}"
        WEIGHTS_PATH_VEH = "models/StixelNExT-Pro_stoic-capybara-555_47.pth"
        WEIGHTS_PATH_TOW = "models/StixelNExT-Pro_wild-vortex-557_15.pth"

        # Models
        vehicle_stixel_predictor = StixelPredictor(weights=WEIGHTS_PATH_VEH)
        tower_stixel_predictor = StixelPredictor(weights=WEIGHTS_PATH_TOW)

        # Dataset
        dataset = cs.Dataloader(DATAPATH)
        # record = dataset[18]
        # frame = record[22]
        # t_cam_vehicle = frame.vehicle.cameras.STEREO_LEFT.info.extrinsic
        # t_vehicle_infra = frame.vehicle.info.extrinsic
        # t_infracam_infra = frame.tower.cameras.VIEW_1.info.extrinsic
        # T_cam_infracam = t_cam_vehicle @ t_vehicle_infra @ np.linalg.inv(t_infracam_infra)
        # print(t_vehicle_infra)
        # print("-------------")
        # print(T_cam_infracam)
        # frames = []

        results = pd.DataFrame(columns=[
            "frame_id", "view", "translation", "rotation-z", "translation_error", "rotation_error", "ransac_fitness", "ransac_rmse", "icp_fitness", "icp_rmse"
        ])

        for datarecord in dataset:
            try:
                for frame in datarecord:
                    for view, infra_cam in frame.tower.cameras:
                        # infra_cam: cs.Camera = frame.tower.cameras.VIEW_1
                        t_cam_vehicle = frame.vehicle.cameras.STEREO_LEFT.info.extrinsic
                        t_vehicle_infra = frame.vehicle.info.extrinsic
                        t_infracam_infra = infra_cam.info.extrinsic
                        T_cam_infracam = t_cam_vehicle @ t_vehicle_infra @ np.linalg.inv(t_infracam_infra)
                        translation_mtx = T_cam_infracam[:3, 3]
                        translation = np.linalg.norm(translation_mtx)
                        yaw_deg = R.from_matrix(T_cam_infracam[:3, :3]).as_euler('xyz', degrees=True)[2]
                        if translation < 30.0 and (-90.0 < yaw_deg < 90.0):
                            # Ground Truth
                            t_cam_vehicle = frame.vehicle.cameras.STEREO_LEFT.info.extrinsic
                            t_vehicle_infra = frame.vehicle.info.extrinsic
                            t_infracam_infra = infra_cam.info.extrinsic
                            T_cam_infracam = t_cam_vehicle @ t_vehicle_infra @ np.linalg.inv(t_infracam_infra)
                            elevation_infra_cam_rad = np.arcsin(t_infracam_infra[2, 2])
                            t_cam_origin = t_vehicle_infra @ t_cam_vehicle
                            elevation_veh_cam_rad = np.arcsin(t_cam_origin[2, 2])
                            # elevation_deg = np.degrees(elevation_veh_cam_rad)
                            # print(f"Elevation-angle (degree): {elevation_deg:.2f}")

                            #frame.tower.cameras.VIEW_2.image.show()
                            #frame.tower.cameras.VIEW_1.image.show()
                            #frame.vehicle.cameras.STEREO_LEFT.image.show()

                            # """
                            prob = 0.5
                            # StixelWorld from Vehicle
                            stixel_veh = vehicle_stixel_predictor.inference(image=frame.vehicle.cameras.STEREO_LEFT.image.image,
                                                                            name=f"{frame.frame_id}_Vehicle",
                                                                            camera_info=frame.vehicle.cameras.STEREO_LEFT.info,
                                                                            prob=prob)
                            # stx.save(stixel_veh, "stixel_veh.ply")
                            #stx_veh_img = stx.draw_stixels_on_image(stixel_veh)
                            #stx_veh_img.show()

                            # StixelWorld from Tower
                            stixel_tow = tower_stixel_predictor.inference(image=infra_cam.image.image,
                                                                          name=f"{frame.frame_id}_Tower",
                                                                          camera_info=infra_cam.info,
                                                                          prob=prob)
                            # stx.save(stixel_tow, "stixel_tow.ply")
                            """
                            stixel_tow_bonus = tower_stixel_predictor.inference(image=frame.tower.cameras.VIEW_2.image.image,
                                                                                name=f"{frame.frame_id}_Tower_bonus",
                                                                                camera_info=frame.tower.cameras.VIEW_2.info,
                                                                                prob=prob)
                            """
                            # stx.save(stixel_tow_bonus, "stixel_tow_bonus.ply")
                            #stx_tow_img = stx.draw_stixels_on_image(stixel_tow)
                            #stx_tow_img.show()
                            # stx.draw_stixels_in_3d(stixel_tow)

                            stx_tow_pts, rgb_tower = stx.convert_to_point_cloud(stixel_tow, slanted_angle_rad=elevation_infra_cam_rad,
                                                                                return_rgb_values=True)     # elevation_infra_cam_rad
                            veh_correction = elevation_veh_cam_rad if np.degrees(elevation_veh_cam_rad) > 2.5 else None
                            stx_veh_pts, rgb_vehicle = stx.convert_to_point_cloud(stixel_veh, slanted_angle_rad=veh_correction,
                                                                                  return_rgb_values=True)
                            pcd_infra = to_pointcloud(stx_tow_pts)
                            pcd_vehicle = to_pointcloud(stx_veh_pts)
                            # stx_tow_bonus_pts, rgb_tower_bonus = stx.convert_to_point_cloud(stixel_tow_bonus,
                            #                                                                 slanted_angle_rad=frame.tower.cameras.VIEW_2.info.extrinsic[2, 2],
                            #                                                                 return_rgb_values=True)
                            # pcd_bonus = to_pointcloud(stx_tow_bonus_pts)
                            # pcd_bonus.colors = o3d.utility.Vector3dVector(rgb_tower_bonus)
                            # """

                            #pcd_vehicle.paint_uniform_color([0.251, 0.878, 0.816])  # vehicle = blueish
                            pcd_infra.paint_uniform_color([1.0, 0.412, 0.706])    # infra = pink

                            # pcd_vehicle.transform(t_vehicle_infra)
                            #o3d.visualization.draw_geometries([pcd_vehicle, pcd_infra])

                            pcd_vehicle.colors = o3d.utility.Vector3dVector(rgb_vehicle)
                            # pcd_infra.colors = o3d.utility.Vector3dVector(rgb_tower)

                            """ Dataset inputs for comparisopn with rich features """
                            # pcd_vehicle = to_pointcloud(frame.vehicle.lidars.TOP.points)
                            # pcd_infra = to_pointcloud(frame.tower.lidars.UPPER_PLATFORM.points)

                            # pcd_vehicle = to_pointcloud(frame.vehicle.lidars.RIGHT.points)
                            # pcd_vehicle.transform(frame.vehicle.lidars.RIGHT.info.extrinsic)
                            # pcd_infra = to_pointcloud(frame.tower.lidars.UPPER_PLATFORM.points)
                            # pcd_infra.transform(frame.tower.lidars.UPPER_PLATFORM.info.extrinsic)

                            voxel_size = 1
                            pcd_vehicle_down, fpfh_vehicle = preprocess_point_cloud(copy.deepcopy(pcd_vehicle), voxel_size)
                            pcd_infra_down, fpfh_infra = preprocess_point_cloud(copy.deepcopy(pcd_infra), voxel_size)

                            """ RANSAC-Matching """
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

                            """ ICP-Matching """
                            result_icp = o3d.pipelines.registration.registration_icp(
                                source=pcd_vehicle,
                                target=pcd_infra,
                                max_correspondence_distance=voxel_size,
                                init=vehicle_tower_tf.mtx,
                                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                            )

                            icp_tf = cs.Transformation('lidar_top', 'lidar_upper_platform', result_icp.transformation)

                            pcd_vehicle.transform(icp_tf.mtx)
                            pred_mtx = icp_tf.mtx
                            gt_mtx = t_vehicle_infra
                            error_mtx = np.linalg.inv(gt_mtx) @ pred_mtx

                            # print(f"Transformation Matrix: \n{pred_mtx}")
                            # print(f"GT Matrix: \n{gt_mtx}")

                            translation = T_cam_infracam[:3, 3]
                            translation_agents = np.linalg.norm(translation)

                            # Translation error
                            translation_error = error_mtx[:3, 3]
                            translation_distance = np.linalg.norm(translation_error)
                            # print("Translation error (in meter):", translation_distance)

                            yaw_deg = R.from_matrix(T_cam_infracam[:3, :3]).as_euler('xyz', degrees=True)[2]
                            #rotation = T_cam_infracam[:3, :3]
                            #rotation_agents_vector = R.from_matrix(rotation).as_rotvec()
                            #rotation_agents_deg = np.linalg.norm(rotation_agents_vector) * 180 / np.pi

                            # Rotation error
                            rotation_matrix = error_mtx[:3, :3]
                            rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()
                            rotation_angle_deg = np.linalg.norm(rotation_vector) * 180 / np.pi

                            results.loc[len(results)] = [
                                frame.frame_id, view, translation_agents, yaw_deg, translation_distance, rotation_angle_deg, ransac_result.fitness, ransac_result.inlier_rmse, result_icp.fitness, result_icp.inlier_rmse
                            ]
                            print(f"{frame.frame_id}, {view}, {translation_agents}, {yaw_deg}, {translation_distance}, {rotation_angle_deg}, {ransac_result.fitness}, {ransac_result.inlier_rmse}, {result_icp.fitness}, {result_icp.inlier_rmse}")
                            # print("Error matrix (GT⁻¹ * Pred):")
                            # print(error_mtx)
                            # print("\nTranslation error (in meter):", translation_distance)
                            # print("Rotation error (in degree):", rotation_angle_deg)

                            # visualization
                            # o3d.visualization.draw_geometries([pcd_vehicle, pcd_infra])
                            # combined_pcd = pcd_vehicle + pcd_infra + pcd_bonus
                            # o3d.io.write_point_cloud("combined_cloud.ply", combined_pcd)

            except Exception as e:
                print(f"Error processing frame: {datarecord.name}")
                print(e)
        results.to_excel(f"evaluation_results_seq_{sequence}.xlsx", index=False)


if __name__ == "__main__":
    main()
