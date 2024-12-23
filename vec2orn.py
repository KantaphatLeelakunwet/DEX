import numpy as np

def vector_to_euler(vector):
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)
    
    # Extract the components
    x, y, z = vector
    
    # Calculate yaw (ψ)
    yaw = np.arctan2(y, x)  # angle in the XY plane from the x-axis
    # Calculate pitch (θ)
    pitch = np.arcsin(-z)    # angle from the XY plane
    # Calculate roll (φ)
    roll = 0  # Assuming no roll for a direction vector

    # Convert radians to degrees if needed
    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    return roll, pitch, yaw

# Example usage
vector = np.array([0, 0, 1])
euler_angles = vector_to_euler(vector)
print("Euler angles (roll, pitch, yaw):", euler_angles)