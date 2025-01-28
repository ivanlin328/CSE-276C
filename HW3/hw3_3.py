import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor


def load_file(files):
    return np.loadtxt(files)

def fit_plane_ransac(pointcloud):
    # Separate the point cloud into 3D coordinates (xi, yi, zi)
    X = pointcloud[:, :2]  # xi, yi as the coordinates
    Y = pointcloud[:, 2]   # zi as the values
    
    # Fit the RANSAC model
    ransac = RANSACRegressor(residual_threshold=0.2, max_trials=100,random_state=0)
    ransac.fit(X, Y)
    
    a, b = ransac.estimator_.coef_
    c = -1.0  
   
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    outlier = ~ransac.inlier_mask_
    return normal, ransac.inlier_mask_, outlier

def fit_wall_ransac(pointcloud,outlier):
     X= pointcloud[outlier][:,[0,2]] 
     Y= pointcloud[outlier][:,1]
     ransac = RANSACRegressor(residual_threshold=0.2, max_trials=100,random_state=0)
     ransac.fit(X, Y)
     
     return ransac.inlier_mask_
    
    

pointcloud_cse=load_file("CSE-1.asc")

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pointcloud_cse)
o3d.visualization.draw_geometries([point_cloud])

plane_params_ground, inlier_mask,outlier= fit_plane_ransac(pointcloud_cse)
filter_data = pointcloud_cse[outlier]
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pointcloud_cse[inlier_mask])
o3d.visualization.draw_geometries([point_cloud])

inlier_mask1=fit_wall_ransac(pointcloud_cse, outlier)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(filter_data[inlier_mask1])
o3d.visualization.draw_geometries([point_cloud])










