#Forward Kinematic

import sympy as sp

# Defined symbolic variables
q0, q1,q2 =sp.symbols('q0 q1 q2')

import sympy as sp

# Define the symbolic variables
q0, q1, q2 = sp.symbols('q0 q1 q2')

# Define the rotation matrix
R = sp.Matrix([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])

# Define the transformation matrix
T0=sp.Matrix([[sp.cos(q0),-sp.sin(q0),5],
             [sp.sin(q0),sp.cos(q0),0],
             [0,0,1]])
T1=sp.Matrix([[sp.cos(q1),-sp.sin(q1),10],
             [sp.sin(q1),sp.cos(q1),0],
             [0,0,1]])
T2=sp.Matrix([[sp.cos(q2),-sp.sin(q2),10],
             [sp.sin(q2),sp.cos(q2),0],
             [0,0,1]])
M=T0*T1*T2
# Define the Z axis vector
E = sp.Matrix([10, 0, 1])

# Perform the matrix multiplication
end_effector_position = R * M * E
# Print the result
sp.pprint(sp.simplify(end_effector_position))

q=sp.Matrix([q0,q1,q2])

#Calculate the Jacobian Matrix
print("\n Jacobian Matrix:")
J=end_effector_position.jacobian(q)

sp.pprint(sp.simplify(J))


# Target position
x = (-10) * sp.sin(q0 + q1 + q2) - 10 * sp.sin(q0 + q1) - 10 * sp.sin(q0)
y = 10 * sp.cos(q0 + q1 + q2) + 10 * sp.cos(q0 + q1) + 10 * sp.cos(q0) + 5
end_effector_position=sp.Matrix([x,y])

p_target = sp.Matrix([10, 15])

q_values = {q0: 0, q1: 0, q2: 0}

while True:
    # The current position is calculated by substituting the joint values (0,0,0) into the end effector position.
    p_current = end_effector_position.subs(q_values)
    
    # The change in position \( \Delta p \) is computed as the difference between the target and current positions.
    delta_p = p_target - p_current  
    
    # Calculate the Jacobian matrix                  
    J = end_effector_position.jacobian([q0, q1, q2])
    
    # The pseudo-inverse of the Jacobian is calculated to determine the joint velocities needed to move towards the target position
    J_1 = J.subs(q_values)
    J_inv = J_1.pinv()
    
    joint_velocity = J_inv * delta_p  # \( dq = j_{inv} \cdot dp \)

    # Update joint angles based on calculated joint velocities
    for i, q in enumerate([q0, q1, q2]):
        q_values[q] += joint_velocity[i]
    
    # Calculate the null space of the Jacobian matrix
    null_space = J_1.nullspace()
    
    # Update joint angles using the null space vector for redundancy resolution
    for i, q in enumerate([q0, q1, q2]):
        q_values[q] += null_space[0][i] * 0.01  
        # Use the null space to adjust joint angles slightly, ensuring smooth motion
    if delta_p.norm() <= 0.001:
        break
print("joint angle configuration",q_values)
print("the position of the end effector",end_effector_position.subs(q_values))
