import numpy as np
from sklearn.linear_model import RANSACRegressor
import open3d as o3d
from sklearn.cluster import KMeans

def load_file(files):
    return np.loadtxt(files)

def fit_plane_ransac(pointcloud):
    # Separate the point cloud into 3D coordinates (xi, yi, zi)
    X = pointcloud[:, :2]  # xi, yi as the coordinates
    Y = pointcloud[:, 2]   # zi as the values
    
    # Fit the RANSAC model
    ransac = RANSACRegressor(min_samples=10,residual_threshold=0.1, max_trials=4000,random_state=0)
    ransac.fit(X, Y)
    
    a, b = ransac.estimator_.coef_
    c = -1.0  
   
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    
    return normal, ransac.inlier_mask_

def cluster_plane_points(pointcloud, inlier_mask):
    # Extract the points that lie on the plane
    plane_points = pointcloud[inlier_mask]

    # Apply K-Means clustering to separate table and non-table points
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(plane_points)
    labels = kmeans.labels_

    # Create separate point clouds for table and non-table
    table_points = plane_points[labels == 0]
    non_table_points = plane_points[labels == 1]

    return table_points, non_table_points
   

pointcloud_object=load_file("TableWithObjects2-1.asc")
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pointcloud_object)
o3d.visualization.draw_geometries([point_cloud])

plane_params_object, inlier_mask2 = fit_plane_ransac(pointcloud_object)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pointcloud_object[inlier_mask2])
o3d.visualization.draw_geometries([point_cloud])

table_points, non_table_points = cluster_plane_points(pointcloud_object, inlier_mask2)
table_cloud = o3d.geometry.PointCloud()
table_cloud.points = o3d.utility.Vector3dVector(table_points)

o3d.visualization.draw_geometries([table_cloud])

print(plane_params_object)
