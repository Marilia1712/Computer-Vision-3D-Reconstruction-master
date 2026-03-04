from turtle import width

import glm
import cv2 as cv
import random
import numpy as np
import os
import glob
import shutil


block_size = 1.0
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


"""
#loop to get background subtraction for each camera

for i in range(1,5):
    video_path_back = f"data/cam{i}/background.avi"
    video_path_fore = f"data/cam{i}/video.avi"
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

    background = background_detection(video_path_back)
    foreground = background_subtraction(video_path_fore, background, kernel)

"""

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


def generate_voxel_grid():

    """
    Create a (full) voxel grid in the world
    """

    voxel_dim = 20.0   # 20 mm (2 cm) ← good compromise speed/quality

    x = np.arange(-640, 640, voxel_dim, dtype=np.float32)
    y = np.arange(0, 640, voxel_dim, dtype=np.float32)
    z = np.arange(-640, 640, voxel_dim, dtype=np.float32)

    grid = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([g.ravel() for g in grid], axis=1)

    print("Voxel grid size:", len(voxels))

    return voxels



def create_lookup_table(voxels, cam_params):
    """
    Vectorized lookup table:
    lut[cam][i] -> (u,v) pixel of voxel i in that camera
    """

    voxels = voxels.astype(np.float32)

    lut = {}

    for cam in range(1, 5):

        camera_matrix, dist, rvec, tvec = cam_params[f'cam{cam}']

        # project ALL voxels at once (fast)
        projected_pixels, _ = cv.projectPoints(
            voxels,
            rvec,
            tvec,
            camera_matrix,
            dist
        )

        # shape (N,1,2) → (N,2)
        projected_pixels = projected_pixels.reshape(-1, 2)

        # store per-camera array
        lut[cam] = projected_pixels.astype(np.int32)

    return lut


def set_voxel_positions(width, height, depth):
    """
    Calculate proper voxel arrays
    """

    colors = [] #TODO: do we need to do anything about the colors?
    frames = []
    background = {1: [], 2: [], 3: [], 4: []}
    foreground = {1: [], 2: [], 3: [], 4: []}
    final_voxel, data = [], []

    #for each camera get background subtraction
    for i in range(1,5):
                video_path_back = f"data/cam{i}/background.avi"
                video_path_fore = f"data/cam{i}/video.avi"
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

                background[i] = background_detection(video_path_back)
                foreground[i] = background_subtraction(video_path_fore, background[i], kernel)

    # 1 define a 3d voxel grid
    voxels = generate_voxel_grid()
    # 2 project the 3d voxel grid to each camera view
    lut = create_lookup_table(voxels, params)

    #for each frame...
    #frames = min(len(foreground[1]), len(foreground[2]), len(foreground[3]), len(foreground[4]))

    frames = 1

    for f in range(frames):

            #for each voxel in the grid...
            for voxel_idx, voxel in enumerate(voxels):
                keep = True

                #for each camera...
                for cam in range(1,5):
                    
                    u, v = lut[cam][voxel_idx]
                            
                    # 3 check if projected pixel lies inside the foreground mask
                    mask = foreground[cam][f]
                    if not (0 <= u < mask.shape[1] and 0 <= v < mask.shape[0]) or mask[v,u] == 0: # not only if its active, must address issues of Out Of Bounds
                        keep = False
                        break

                # 4 collect all voxels that are active in all camera views
                if keep:
                    final_voxel.append(voxel)

    scale = 0.1  # mm → cm
    data = [[v[0]*scale, v[1]*scale, v[2]*scale] for v in final_voxel]

    colors = [[1.0, 1.0, 1.0] for _ in final_voxel]

    print("Foreground nonzero:", np.count_nonzero(foreground[1][0]))
    print("Num voxels kept:", len(final_voxel))

    return data, colors


"""
Referencing image: https://uu.brightspace.com/content/enforced/44949-BETA--2025--3-GS--INFOMCV--V/csfiles/home_dir/courses/BETA-2023-3-GS-INFOMCV-V/BETA-2023-3-GS-INFOMCV-V/flow.png

-   Now we have a function to get a set of voxels visible in all cameras (active voxels).
    (TODO: for now it loops through 4 sample frames but it should be adapted to loop through all frames in the video)

-   NOTE: when choosing the parameters to call the function, choose them wisely so that the voxels are positioned nicely (not too small/big, etc.)
    This was something Mr.Boobs stressed on in the lecture

-   The output of the function is a set of voxels that are active in all camera views, which can be used for 3D reconstruction or visualization.

-   TODO: we need to give the 3d array of turned on voxels to the visualization code

"""