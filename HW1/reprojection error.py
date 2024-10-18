import numpy as np
import matplotlib.pyplot as plt

# Define constants for the checkerboard dimensions
square_size = 25
num_corner_x = 8
num_corner_y = 6

def load_file(file_path):
    """Load image points from a text file. Assumes each image has the same number of points."""
    data = np.loadtxt(file_path)
    data = np.reshape(data, [9, 48, 2])  # Assuming 9 images, 48 points per image
    return data

def construct_world():
    """Construct the world coordinates for the checkerboard corners."""
    world_points = []
    for i in range(num_corner_y):
        for j in range(num_corner_x):          
            world_points.append([j * square_size, i * square_size, 1])          
    return np.array(world_points)

def matrix_m(world_points, image_points):
    """Construct matrix M for homography calculation."""
    M = []
    for (X, Y, _), (u, v) in zip(world_points, image_points):
        M.append([-X, -Y, -1, 0, 0, 0, u*X, u*Y, u])
        M.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
    return np.array(M)

def homography(M):
    """Calculate the homography matrix from matrix M."""
    U, S, Vt = np.linalg.svd(M)
    h = Vt[-1] 
    H = h.reshape(3, 3)
    return H

def construct_L(H):
    h = H.flatten()

    v_12 = np.array([
        h[0] * h[1],
        h[0] * h[4] + h[3] * h[1],
        h[3] * h[4],
        h[6] * h[1] + h[0] * h[7],
        h[6] * h[4] + h[3] * h[7],
        h[6] * h[7]
    ])

    v_11 = np.array([
        h[1] * h[1],
        h[1] * h[4] + h[4] * h[1],
        h[4] * h[4],
        h[7] * h[1] + h[1] * h[7],
        h[7] * h[4] + h[4] * h[7],
        h[7] * h[7]
    ])

    v_00 = np.array([
        h[0] * h[0],
        h[0] * h[3] + h[3] * h[0],
        h[3] * h[3],
        h[6] * h[0] + h[0] * h[6],
        h[6] * h[3] + h[3] * h[6],
        h[6] * h[6]
    ])

    # Stack the rows to form L_i
    L_i = np.vstack([v_12.T, (v_00.T - v_11.T)])
    return L_i

def find_best_fit_b(L_i):
    """Perform SVD on matrix L and return the last row of Vt (smallest singular value vector)."""
    U, S, Vt = np.linalg.svd(L_i)
    b_best_fit = Vt[-1]  # The last row of Vt corresponds to the smallest singular value
    return b_best_fit
def compute_reprojection_error(H, world_points, image_points):
    """Calculate the total reprojection error."""
    total=0
    for (X, Y, _), (u, v) in zip(world_points, image_points):
        
        # Project the world point using the homography matrix
        projected_point = H @ np.array([X, Y, 1])  
        normalize=projected_point[:2]/projected_point[2]
        # Calculate reprojection error
        error = np.sqrt((normalize[0] - u) ** 2 + (normalize[1] - v) ** 2)   
        total+=error   
    return total 
def matrix_A(b_best_fit):
    B11, B12, B22, B13, B23, B33 = b_best_fit

    # Step 1: Calculate v0
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)

    # Step 2: Calculate λ (lambda)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    # Step 3: Calculate alpha
    alpha = np.sqrt(lambda_ / B11)

    # Step 4: Calculate beta
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))

    # Step 5: Calculate gamma
    gamma = -B12 * alpha**2 * beta / lambda_

    # Step 6: Calculate u0
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_

    # Return the intrinsic matrix A
    A = np.array([
    [alpha, gamma, u0],
    [0, beta, v0],
    [0, 0, 1]
    ])
    return A

def extrinsic_matrix(H, A):
    h0 = H[:, 0]
    h1 = H[:, 1]
    h2 = H[:, 2]

   
    lambda_ = 1 / np.linalg.norm(np.dot(np.linalg.inv(A), h1))

    
    r0 = lambda_ * np.dot(np.linalg.inv(A), h0)
    r1 = lambda_ * np.dot(np.linalg.inv(A), h1)
    t = lambda_ * np.dot(np.linalg.inv(A), h2)

    
    r2 = np.cross(r0, r1)  

    
    R_T = np.column_stack((r0, r1, r2))
    extrinsic = np.column_stack((R_T, t))  

    return extrinsic

def trajectory(extrinsic):
    R= extrinsic[:, :3]  
    t = extrinsic[:, 3]   
    C = -np.dot(R.T, t)
    return C
    


def main():
    file_path = "/Users/ivanlin328/Desktop/imgpoints.txt"
    image = load_file(file_path)  # Load image points
    world_points = construct_world()  # Construct world points for the checkerboard
    error=0
    L = []  # To stack the L_i matrices
     
    for i in range(len(image)):  
        image_points = image[i]  # Get image points for the i-th image
        M = matrix_m(world_points, image_points)  # Compute matrix M
        H = homography(M)  # Compute homography matrix H
        #print(H)
        L_i = construct_L(H)  # Compute the corresponding L_i for the homography
        L.append(L_i)  # Stack it to form matrix L
    L = np.vstack(L)  # Stack all L_i matrices to form the final L
    b_best_fit = find_best_fit_b(L)
    A= matrix_A(b_best_fit)
    trajectory_points = []
    for i in range(len(image)):  
        image_points = image[i]  # Get image points for the i-th image
        M = matrix_m(world_points, image_points)  # Compute matrix M
        H = homography(M)  # Compute homography matrix H
        #print(H)
        L_i = construct_L(H)   
        E=extrinsic_matrix(H,A)
        print(E) 
        C = trajectory(E)
        trajectory_points.append(C)
        total_error = compute_reprojection_error(H, world_points, image_points)
        error+=total_error
        #print(C)   
    #print("Total Reprojection Error:", error)
    # Find the best fit for b by solving Lb = 0 using SVD
    #print("Best fit for b:", b_best_fit)
    #print(A)
    
    trajectory_points=np.array(trajectory_points)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], marker='o', linestyle='-', color='b')
    plt.title("Camera Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.axis('equal')  
    plt.show()
    
if __name__ == "__main__":
    main()



        
   

