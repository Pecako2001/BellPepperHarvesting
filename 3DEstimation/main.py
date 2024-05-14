import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_ply_model(filepath):
    return o3d.io.read_point_cloud(filepath)

def visualize_model(model, transformation_matrix=np.eye(4)):
    # Apply the transformation to the model
    model.transform(transformation_matrix)
    
    # Visualize the model
    o3d.visualization.draw_geometries([model])

def create_transformation_matrix(angle_degrees, axis='z'):
    angle_radians = np.deg2rad(angle_degrees)
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0, 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    else:  # axis 'y'
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians), 0],
            [0, 0, 0, 1]
        ])
    return rotation_matrix

# Load the model
model = load_ply_model("bell_pepper.ply")

# Create a transformation matrix to simulate the model being rotated 30 degrees around the Z-axis
transformation_matrix = create_transformation_matrix(30, 'z')

# Visualize the model with the applied transformation
visualize_model(model, transformation_matrix)
