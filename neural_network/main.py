import math
import pandas
from keras.models import load_model
import keras
import numpy as np
from PIL import Image
import cv2

### VECTOR MANIPULATION FUNCTIONS ###
# Functions for taking raw data vectors (head position, facing/eye direction) and find where the user is looking

# generates a scalar s such that the magnitude of (pos + s*dir) == rad
# allows us to find a point on a sphere from a direction and a point inside the sphere
# pos: offset from center, 3 dimensional vector, magnitude <= rad
# dir: direction toward sphere, 3 dimensional vector
# rad: radius of sphere, scalar
# post: returns s
def theta_scalar(pos, dir, rad=1):
    # user on sphere
    if pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] == rad:
        return 0
    
    # polynomial coefficents
    a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]
    b = 2*(pos[0]*dir[0] + pos[1]*dir[1] + pos[2]*dir[2])
    c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - rad*rad

    th_s = [(-b + math.sqrt(b*b - 4*a*c))/2/a, (-b - math.sqrt(b*b - 4*a*c))/2/a]
    
    return max(th_s)

# adds two 3-dimensional vectors together
# post: returns the sum of vec1, vec2
def vec_add(vec1, vec2):
    return [vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]]

# converts a 3D vector from cartesian to spherical coordinates
# vec: 3 dimensional vector in cartesian
# post: returns [theta, phi, r] (inclination, azimuth, radius)
# as we are working with a sphere of radius 1, these should also be our latitude, longitude for locating points in the videos
def to_spherical(vec):
    # exception handling
    # return a zero vector if a zero vector is given
    if vec[0] == 0 and vec[1] == 0 and vec[3] == 0:
        return [0,0,0]
    
    # radius
    r = vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]

    # azimuth phi
    ph = math.atan2(vec[1], vec[0])

    # inclination theta
    th = math.atan2(vec[2], math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]))

    return [th, ph, r]


### END VECTOR MANIPULATION ###

### IMAGE MANIPULATION FUNCTIONS ###
## Adapted from ChatGPT code
def crop_image_around_point(image_path, x, y, crop_size=(100, 100)):
    """
    Crop an image around a specified point.

    Args:
        image (PIL.Image): The input image.
        x (int): X-coordinate of the center point.
        y (int): Y-coordinate of the center point.
        crop_size (tuple): Size of the cropped region (width, height). Default is (100, 100).

    Returns:
        PIL.Image: The cropped image.

    Raises:
        ValueError: If the specified crop size is larger than the image size.
    """
    with Image.open(".\\vid_frames\\uncropped\\" + image_path) as image:
        image.load()
    # Convert image to numpy array for easier manipulation
    img_array = np.array(image)

    # Get image dimensions
    img_width, img_height = image.size

    # Calculate crop boundaries
    half_width = crop_size[0] // 2
    half_height = crop_size[1] // 2
    x1 = max(0, x - half_width)
    y1 = max(0, y - half_height)
    x2 = min(img_width, x + half_width)
    y2 = min(img_height, y + half_height)

    # Check if crop size is valid
    if (x2 - x1) != crop_size[0] or (y2 - y1) != crop_size[1]:
        raise ValueError("Crop size exceeds image dimensions.")

    # Crop the image
    cropped_image = Image.fromarray(img_array[y1:y2, x1:x2])

    cropped_image.save(".\\vid_frames\\cropped\\" + image_path, format= "JPEG")

# adapted from https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
##  solution by: beebee8 and Arsen Khachaturyan
# extract .jpg files to directory `pathOut` from file `pathIn`
# pathIn: path to video, includes video
# pathOut: name of folder to save to. Should correlate to the video
def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        # adjust framerate to match sample rate from VR sim
        framerate = 60

        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000/framerate))
        success,image = vidcap.read()
        # print ('Read a new frame: ', success)
        # save selected frames
        cv2.imwrite(".\\vid_frames\\uncropped\\" + pathOut + "\\frame%d.jpg" % count, image)
        count = count + 1
### END IMAGE MANIPULATION ###

### IMAGE PREDICTION/CLASSIFICATION ###
# load an image from a path `img_file` and predict the likelihood of it being a face using face_detection.keras
# returns a value [0,1] with the predicted probability of the given image being a face
def predictImage(img_file):
    model = load_model('face_detection.keras')

    img = keras.utils.load_img(img_file, target_size= (180, 180))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis= 0)

    predictions = model.predict(x)

    score = float(keras.activations.sigmoid(predictions[0][0]))
    return score
    # debug statement
    print(f"This image is {100 * (1 - score):.2f}% distractor and {100 * score:.2f}% face.")
### END IMAGE CLASSIFICATION ###

### TODO: ###
# For video in the simulation (./vr_videos), run extractImages, extracting the images to separate folders
# Use pandas to read .xlsx of data collected in vr
# Convert vectors to positions using vector manipulation functions
# crop images around the point found via vector manipulation
# run predictImage on cropped images
### ~   ~ ###