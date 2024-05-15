import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import open3d as o3d

"""
This script demonstrates how to estimate the 3D pose of a green bell pepper using YOLOv8 segmentation and Open3D.
"""

# Load the YOLOv8 segmentation model
model = YOLO('SegBest.pt')

# Load the image
image_path = 'TestImages/3D_BLP_034.jpg'
image = cv2.imread(image_path)

# Run YOLOv8 inference on the image
results = model(image, imgsz=320, iou=0.2, conf=0.20)

# Visualize the results on the image
annotated_image = results[0].plot()

# Display the annotated image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.title('YOLOv8 Inference')
plt.axis('off')
plt.show()

# Load the 3D model
pcd = o3d.io.read_point_cloud("PoseEstimation/bell_pepper.ply")
points_3d = np.asarray(pcd.points)

# Process each detection result
for r in results:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # Iterate each object contour
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]

        # Debug: Check number of masks and current index
        print(f"Processing object {ci+1}/{len(r)}: Label={label}")
        print(f"Number of masks: {len(c.masks.xy)}")

        if len(c.masks.xy) <= ci:
            print(f"Skipping object {ci} due to insufficient masks.")
            continue

        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy[ci].astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Find contours in the mask
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assuming the largest contour corresponds to the bell pepper
            contour = max(contours, key=cv2.contourArea)

            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)

            # Draw the ellipse on the original image
            image_with_ellipse = image.copy()
            cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)

            # Calculate the minimum enclosing rotated rectangle (bounding box)
            rotated_rect = cv2.minAreaRect(contour)
            box_points = cv2.boxPoints(rotated_rect)
            box_points = np.intp(box_points)

            # Draw the bounding box on the original image
            cv2.drawContours(image_with_ellipse, [box_points], 0, (0, 0, 255), 2)

            # Draw an arrow pointing to the peduncle of the bell pepper
            peduncle_point = (int(ellipse[0][0]), int(ellipse[0][1] - ellipse[1][1] / 2))
            cv2.arrowedLine(image_with_ellipse, peduncle_point, (peduncle_point[0], peduncle_point[1] - 50), (255, 0, 0), 2, tipLength=0.3)

            # Get the 2D points of the contour
            points_2d = np.array(contour).reshape(-1, 2)

            # Estimate the camera intrinsic parameters (assuming pinhole camera model)
            fx, fy = 800, 800  # Focal lengths
            cx, cy = image.shape[1] // 2, image.shape[0] // 2  # Principal point

            # Project the 3D points to 2D
            def project_points(points, K):
                points = points[:, :3]  # Remove the homogeneous coordinate
                points_2d = np.dot(K, points.T).T
                points_2d = points_2d[:, :2] / points_2d[:, 2, np.newaxis]
                return points_2d

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            points_2d_proj = project_points(points_3d, K)

            # Subsample the projected points to match the number of 2D contour points
            if len(points_2d_proj) > len(points_2d):
                idx = np.random.choice(len(points_2d_proj), len(points_2d), replace=False)
                points_2d_proj = points_2d_proj[idx]

            # Use ICP to align the 2D points with the projected 3D points
            def icp_2d(source, target, max_iterations=100, tolerance=1e-6):
                src_mean = np.mean(source, axis=0)
                tgt_mean = np.mean(target, axis=0)
                src_centered = source - src_mean
                tgt_centered = target - tgt_mean
                W = np.dot(src_centered.T, tgt_centered)
                U, _, VT = np.linalg.svd(W)
                R = np.dot(VT.T, U.T)
                t = tgt_mean - np.dot(src_mean, R.T)
                return R, t

            # Perform ICP alignment (in 3D)
            def icp_3d(source, target, max_iterations=100, tolerance=1e-6):
                src_mean = np.mean(source, axis=0)
                tgt_mean = np.mean(target, axis=0)
                src_centered = source - src_mean
                tgt_centered = target - tgt_mean
                H = np.dot(src_centered.T, tgt_centered)
                U, S, VT = np.linalg.svd(H)
                R = np.dot(VT.T, U.T)
                if np.linalg.det(R) < 0:
                    VT[2, :] *= -1
                    R = np.dot(VT.T, U.T)
                t = tgt_mean - np.dot(src_mean, R)
                return R, t

            # Perform ICP alignment
            R, t = icp_3d(points_3d, points_3d)

            # Apply the transformation to the 3D model
            pcd_transformed = np.dot(points_3d, R.T) + t

            # Define the origin and the axes in 3D space
            origin = np.mean(pcd_transformed, axis=0)
            axis_length = 0.1  # Adjust based on the scale of your model

            x_axis = origin + axis_length * np.array([1, 0, 0])
            y_axis = origin + axis_length * np.array([0, 1, 0])
            z_axis = origin + axis_length * np.array([0, 0, 1])

            # Project the axes to 2D
            def project_axis_point(point, K):
                point_homogeneous = np.hstack((point, 1))  # Convert to homogeneous coordinates
                projected = np.dot(K, point_homogeneous[:3])
                return projected[:2] / projected[2]

            origin_2d = project_axis_point(origin, K)
            x_axis_2d = project_axis_point(x_axis, K)
            y_axis_2d = project_axis_point(y_axis, K)
            z_axis_2d = project_axis_point(z_axis, K)

            # Draw the 3D axes on the 2D image
            def draw_axis(image, start_point, end_point, color):
                start_point = tuple(start_point.astype(int))
                end_point = tuple(end_point.astype(int))
                cv2.arrowedLine(image, start_point, end_point, color, 2, tipLength=0.3)

            draw_axis(image_with_ellipse, origin_2d, x_axis_2d, (0, 0, 255))  # X-axis in red
            draw_axis(image_with_ellipse, origin_2d, y_axis_2d, (0, 255, 0))  # Y-axis in green
            draw_axis(image_with_ellipse, origin_2d, z_axis_2d, (255, 0, 0))  # Z-axis in blue

            # Display the result
            image_with_ellipse_rgb = cv2.cvtColor(image_with_ellipse, cv2.COLOR_BGR2RGB)
            plt.imshow(image_with_ellipse_rgb)
            plt.title('Pose Estimation of Green Bell Pepper with 3D Axes')
            plt.axis('off')
            plt.show()

            # Extract pose information
            center = rotated_rect[0]
            size = rotated_rect[1]
            angle = rotated_rect[2]

            # Print pose information
            print(f"Center: {center}")
            print(f"Size: {size}")
            print(f"Angle: {angle}")
else:
    print("No masks found in the image.")
