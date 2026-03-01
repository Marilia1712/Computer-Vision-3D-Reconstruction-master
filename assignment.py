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
   
def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
    return data, colors

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations


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



