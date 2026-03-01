from turtle import width

import glm
import cv2 as cv
import random
import numpy as np
import os
import glob
import shutil


block_size = 10
def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def get_cam_positions():
    """
    Generates camera locations at the 4 corners
    of the room based on the locations estimated in Task 1
    """
    cam_pos_list = []

    for i in range(1,5):
        cam_pos_list.append([cam_pos[f'cam{i}'][0], cam_pos[f'cam{i}'][1], cam_pos[f'cam{i}'][2]]) 

    return [cam_pos_list, \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]]

def get_cam_rotation_matrices():
    """"
    Generates camera rotation matrices based on the
    orientations estimated in Task 1
    """
    cam_rot_list = []

    for i in range(1,5):
        R = cam_R[f'cam{i}']

        #4x4 matrix is needed for rendering
        mat4= np.eye(4)
        mat4[:3,:3] = R
        cam_rot_list.append(mat4)
    
    return cam_rot_list


"""
    Exercise 2
"""

def background_detection(video_path):
    back = cv.VideoCapture(video_path)
    n_frames = back.get(cv.CAP_PROP_FRAME_COUNT)
    print("Number of frames: ", n_frames)

    #take half of the frames
    frames_indeces = np.linspace(0, n_frames, int(n_frames//2), dtype = np.int32)
    print("Frames considered: ", len(frames_indeces))

    frames_back = []

    #Preprocess all frames
    for index in frames_indeces:
        back.set(cv.CAP_PROP_POS_FRAMES, index)
        ret, frame = back.read()
        if not ret:
            continue
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frames_back.append(frame_hsv)

    all_frames_back = np.stack(frames_back, axis = 0)

    #Take median
    background = np.median(all_frames_back, axis = 0)
    background = background.astype(np.uint8)
    cv.imshow("Background", background)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return background


def background_subtraction(video_path, background, kernel, k = 3):
    fore = cv.VideoCapture(video_path)

    foreground_mask = []
    n_frames_fore = int(fore.get(cv.CAP_PROP_FRAME_COUNT))
    i = 0

    while i < n_frames_fore:
        ret, frame = fore.read()
        if not ret:
            i += 1
            continue

        fore_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        hue_diff = cv.absdiff(fore_hsv[:,:,0], background[:,:,0])
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)   

        sat_diff = cv.absdiff(fore_hsv[:,:,1], background[:,:,1])
        val_diff = cv.absdiff(fore_hsv[:,:,2], background[:,:,2])
        

        # TODO: build stronger automation for auto threshold detection
        mean_h = np.mean(hue_diff)
        std_h = np.std(hue_diff)

        mean_s = np.mean(sat_diff)
        std_s = np.std(sat_diff)

        mean_v = np.mean(val_diff)
        std_v = np.std(val_diff)

        thresh_hue = mean_h + k * std_h
        thresh_sat = mean_s + k * std_s
        thresh_val = mean_v + k * std_v

        _, mask_h = cv.threshold(hue_diff, thresh_hue, 255, cv.THRESH_BINARY)
        _, mask_s = cv.threshold(sat_diff, thresh_sat, 255, cv.THRESH_BINARY)
        _, mask_v = cv.threshold(val_diff, thresh_val, 255, cv.THRESH_BINARY)

        mask = cv.bitwise_or(mask_h, mask_s)
        mask = cv.bitwise_or(mask, mask_v)

        mask = cv.erode(mask, kernel, iterations=1)
        mask = cv.dilate(mask, kernel, iterations=1)   

        mask = cv.dilate(mask, kernel, iterations=1)
        mask = cv.erode(mask, kernel, iterations=1)   
        foreground_mask.append(mask)

        cv.imshow("Hue diff", hue_diff)
        cv.imshow("Sat diff", sat_diff)
        cv.imshow("Val diff", val_diff)
        cv.imshow("Final Mask", mask)

        key = cv.waitKey(1) & 0xFF        
        if key == 27 or key == ord('q'):
            break

        i += 1

    fore.release()
    cv.destroyAllWindows()
    return foreground_mask


#loop to get background subtraction for each camera

for i in range(1,5):
    video_path_back = f"data/cam{i}/background.avi"
    video_path_fore = f"data/cam{i}/video.avi"
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    background = background_detection(video_path_back)
    foreground = background_subtraction(video_path_fore, background, kernel)

"""
Exercise 3
"""

# Get camera position (adaptation of code from Assingnment 1)

cam_pos = {}
cam_orient = {}
cam_R = {}

rvecs = {}
tvecs = {}
cam_matrix = {}
dist_coeff = {}
params = {}

for i in range(1, 5):

    fs = cv.FileStorage(f"data/cam{i}/config.xml", cv.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist = fs.getNode("DistortionCoeffs").mat()
    fs.release()

    rvec = rvec.reshape(3,1)
    tvec = tvec.reshape(3,1)
    dist = dist.reshape(5,1)
    camera_matrix = camera_matrix.reshape(3,3)
    rvecs[f'cam{i}'] = rvec
    tvecs[f'cam{i}'] = tvec
    cam_matrix[f"cam{i}"] = camera_matrix
    dist_coeff[f"cam{i}"] = dist 

    R, _ = cv.Rodrigues(rvec)

    cam_position = (-R.T @ tvec).reshape(3,)

    cam_direction = (R.T @ np.array([0, 0, 1.0])).reshape(3,)
    cam_direction = cam_direction / np.linalg.norm(cam_direction)

    cam_pos[f"cam{i}"] = cam_position
    cam_orient[f"cam{i}"] = cam_direction
    cam_R[f"cam{i}"] = R

    #Store params for each camera
    params[f'cam{i}'] = [camera_matrix, dist, rvec, tvec]


def generate_voxel_grid(width, height, depth):
    """
    Generates a 3D voxel grid in the room
    Same logic as generate_grid but with 3D coordinates
    """
    grid_data, colors = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                grid_data.append([x*block_size - width/2, y*block_size - height/2, z*block_size - depth/2])
                colors.append([1.0, 1.0, 1.0] if (x+y+z) % 3 == 0 else [0, 0, 0])
    return grid_data, colors


def set_voxel_positions(width, height, depth):
    """
    Calculate proper voxel arrays
    """

    data, colors = [], []
    frames = []

    #for each frame...
    for f in frames:
        # 1 define a 3d voxel grid
            grid_data, colors = generate_voxel_grid(width, height, depth)
           
        # 2 project the 3d voxel grid to each camera view
            #use calibrate camera parameters (and functions)
            #compute pixel coordinates for voxel in each camera

            #for each voxel...
            projected_pixels = {1: [], 2: [], 3: [], 4: []}

            for voxel in grid_data:
                #for each camera...
                for cam in range(1,5):
                    #use calibrate camera parameters...
                    camera_matrix, dist, rvec, tvec = params[f'cam{cam}']
                    #compute pixel coordinates (project voxel-> pixel(u,v))
                    projected_pixel, _ = cv.projectPoints(voxel, rvec, tvec, camera_matrix, dist)
                    projected_pixels[cam].append(projected_pixel[0][0])
                            
        # 3 check visibility against silhouttes
            #for each camera check if projected pixel lies inside the foreground mask
            #if its not in all four camera masks, discard it

        # 4 collect all voxels
    
    return data, colors