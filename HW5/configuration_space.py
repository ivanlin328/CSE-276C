import numpy as np  
import matplotlib.pyplot as plt  
from shapely.geometry import LineString, Polygon, Point 
from shapely.affinity import rotate, translate  

# Define the simplified world and robot
WORLD_WIDTH = 400  #
WORLD_HEIGHT = 150  
OBSTACLES = [# Define a list of obstacles
    LineString([(50,100),(150,100)]),  
    LineString([(100,0), (100, 100)]),  
    LineString([(200,50), (200, 150)]),
    LineString([(250,100),(350,100)]),
    LineString([(300,0),(300,100)]),
    LineString([(0,0),(0,150)]),  # Left wall
    LineString([(0,0),(400,0)]),  # Top wall
    LineString([(400,0),(400,150)]), # Right wall
    LineString([(0,150),(400,150)])  # Bottom wall
]

ROBOT_RADIUS = 18 # Define the radius of the circular robot
# Generate a circular robot using a buffer around a point
ROBOT_circle = Point(0, 0).buffer(ROBOT_RADIUS)

ROBOT_SIZE = 25# Define the size of the robot
ROBOT = Polygon([(-ROBOT_SIZE/2, -ROBOT_SIZE/2),  # Define the shape of the robot (a square)
                  (ROBOT_SIZE/2, -ROBOT_SIZE/2),
                  (ROBOT_SIZE/2, ROBOT_SIZE/2),
                  (-ROBOT_SIZE/2, ROBOT_SIZE/2)])

# Calculate Minkowski sum
def minkowski_sum(obstacle, robot):  # Define the minkowski_sum function to calculate the Minkowski sum of an obstacle and robot
    robot_points = np.array(robot.exterior.coords)  # Get the coordinates of the robot polygon
    obstacle_points = np.array(obstacle.coords)  # Get the coordinates of the obstacle polygon
    minkowski_points = []  # Initialize a list to hold the Minkowski sum points


    for rp in robot_points:  # Iterate over each point of the robot
        for op in obstacle_points:  # Iterate over each point of the obstacle
            minkowski_points.append(rp + op)  # Add the robot's point and obstacle's point to get Minkowski sum points


    minkowski_poly = Polygon(minkowski_points).convex_hull  # Construct a polygon from the Minkowski sum points and compute its convex hull
    return minkowski_poly  # Return the Minkowski sum polygon

# Generate configuration space
def generate_cspace(obstacles, robot, angle_resolution=5):  # Define the generate_cspace function to generate the configuration space
    cspaces = []  # Initialize a list to hold the configuration spaces
    for angle in range(0, 180, angle_resolution):  # Iterate over angles from 0 to 180 with the specified resolution
        rotated_robot = rotate(robot, angle, origin=(0,0), use_radians=False)  # Rotate the robot by the angle
        cspace_polys = [minkowski_sum(ob, rotated_robot) for ob in obstacles]  # Calculate Minkowski sum between the robot and each obstacle
        cspaces.append((angle, cspace_polys))  # Append the angle and corresponding configuration space polygons to the list
    return cspaces  # Return the list of configuration spaces

# Plotting function to visualize configuration spaces
def plot_cspace(angle,cspaces, world_width, world_height):  # Define the plot_cspace function to plot the configuration space
        plt.figure(figsize=(13, 5))
        plt.title(f"C-Space at {angle}°")  # Set the title of each subplot
        plt.xlim(0, world_width)  # Set the x-axis range
        plt.ylim(world_height,0 )  # Set the y-axis range
        
        # Set the x and y axis ticks
        x_ticks = np.arange(0, world_width + 10, 10)
        y_ticks = np.arange(0, world_height + 10, 10)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        # Format the tick labels
        plt.gca().xaxis.set_ticks_position('top')  # Set the tick positions of the x-axis to the top
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))

        plt.tick_params(axis='both', which='major', labelsize=8)  # Set the font size of the tick labels
        plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)  # Draw grid lines

        for ob in OBSTACLES:  # Iterate over obstacles
            x, y = ob.xy  # Get the x and y coordinates of the obstacle
            plt.plot(x, y, color='gray', linewidth=2)  # Plot the obstacle in gray

        for cspace_poly in cspaces[angle // 5][1]:  # Iterate over the configuration space polygons for the current angle
            x, y = cspace_poly.exterior.xy  # Get the x and y coordinates of the configuration space polygon
            plt.fill(x, y, color='blue', alpha=0.3)  # Plot the configuration space in blue with transparency

        plt.tight_layout()  # Adjust the layout of subplots to avoid overlapping
        plt.show()  # Display the plot

# Main program
cspace1=generate_cspace(OBSTACLES, ROBOT_circle)
#for angle in range(0, 180,5):
#    plot_cspace(angle,cspace1, WORLD_WIDTH, WORLD_HEIGHT)
cspaces2 = generate_cspace(OBSTACLES, ROBOT)  # Generate the configuration space
for angle in [0, 45, 90]:
    plot_cspace(angle,cspaces2, WORLD_WIDTH, WORLD_HEIGHT)  # Plot the configuration space
