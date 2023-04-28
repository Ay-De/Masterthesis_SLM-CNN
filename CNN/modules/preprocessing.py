import pandas as pd
import random
import tensorflow as tf
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image_paths_labels(data_directory):
    """
    This functions loads the csv file containing the images and labels from each printed project
    and returns 3 lists containing all images, binary and multiclass labels of the dataset
    """
    image_paths = []
    image_labels_binary = []
    image_labels_multiclass = []

    project_list = glob.glob(data_directory + '**/**/**/image_label.csv')
    
    for n in range(0, len(project_list)):
        dataframe = pd.read_csv(project_list[n], sep=';')
       # dataframe = dataframe[dataframe['Drop_Image'] != 1]

        project_images = list(dataframe['Image'])
        project_path = project_list[n].replace('logs\\image_label.csv', 'logs\\exposures\\')
        
        project_images = [project_path + s for s in project_images]
        project_labels_binary = list(dataframe['Label_Binary'])
        project_labels_multiclass = list(dataframe['Label_Multiclass_V2'])

        image_paths = image_paths + project_images
        image_labels_binary = image_labels_binary + project_labels_binary
        image_labels_multiclass = image_labels_multiclass + project_labels_multiclass

    return image_paths, image_labels_binary, image_labels_multiclass

#Function to unwarp an given image given the 4 corner points and 4 target points (size of output image)
def _unwarp(image, corner_points, powderbed_height, powderbed_width):
    '''Source, see:
    https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    '''

    #unpack the corner_points array into variables for each corner
    (top_left, top_right, bottom_right, bottom_left) = corner_points

    if (powderbed_height == 0) & (powderbed_width == 0):
        #With Pythagoras: Calculate the length of the 'parallel' top and bottom line
        width_bottom = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_top = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        #The longest line 'wins' -> New width of the unwarped powderbed.
        max_width = max(int(width_bottom), int(width_top))

        #With Pythagoras: Calculate the length of the 'parallel' left and right line
        height_right = np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))
        height_left = np.sqrt(((bottom_left[0] - top_left[0]) ** 2) + ((bottom_left[1] - top_left[1]) ** 2))
        #The longest line 'wins' -> New height of the unwarped powderbed.
        max_height = max(int(height_right), int(height_left))
    
    else:
        max_width = powderbed_width
        max_height = powderbed_height
    
    #Perspective transformation required a transformation matrix.
    #This transformation matrix can be calculated if four points in the source image are known
    #and four corresponding points in the unwarped image.
    #Typically: Warped image (Source): four corner points
    #           Unwarped image (Target): four corner points of the top down view
    #Create an array with the corner points of the target image (rect)
    target_points = np.array([[0, 0],                           #Top left corner
		                    [max_width - 1, 0],                 #Top right corner
		                    [max_width - 1, max_height - 1],    #Bottom right corner
		                    [0, max_height - 1]],               #Bottom left corner
                            dtype=np.float32)

  
    #Calculate the transformation matrix
    M = cv2.getPerspectiveTransform(corner_points, target_points)

    #Apply the transformation matrix and unwarp the image
    unwarped_image = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LINEAR)

    return unwarped_image


def powderbed_detector(project_directory, powderbed_height=0, powderbed_width=0):
    """
    This functions searches for the powderbed in a given project directory and unwarps it afterwards.
    Original image files will be replaced.
    
    Params:
    project_directory: Path to the printed project.
    i.e. C:\\3D_Druck_Datensatz\\Training_Data\\Daten_2018\\2018-06-28_093817\\
    """
    #Read the CSV file containing the image file names and labels
    dataframe = pd.read_csv(project_directory + 'logs\\image_label.csv', sep=';')
    project_directory = project_directory + 'logs\\exposures\\'
    image_list = list(project_directory + dataframe['Image'])

    #stack the first 10 pictures over each other to make the edges of the powderbed clearer
    for i in range(0, 10):
        image = tf.io.read_file(image_list[i])
        image = tf.image.decode_png(image)
        image = tf.image.rgb_to_grayscale(image).numpy()
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.Canny(image, (255/3), 255)

        if i == 0:
           image_stack = image
        else:
           image_stack = np.dstack((image_stack, image))


    #Calculate the average image of the image stack
    image_stack_average = np.mean(image_stack, axis=2, dtype=np.uint8)

    #This kernel contains the shape of the edge to detect (Yes i know, big no no)
    kernel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                   ], dtype=np.uint8)


    #Count the number of true elements (!=0) in the kernel
    kernel_nonzero_count = np.count_nonzero(kernel)
    #Threshold (in %)
    threshold = 0.6

    #Array to store the coordinates of the corners
    corner_points = np.zeros((4, 2), dtype=np.float32)

    #4 Corners -> Range: 4
    for i in range(0, 4):

        #Save the approximate area in which the corner has to be located
        #First loop, top left corner
        if i == 0:
           image_patch = image_stack_average[80:250, 180:350]
        
        #Second loop, top right corner
        elif i == 1:
           image_patch = image_stack_average[80:250, 880:1050]

        #Third loop, bottom right corner
        elif i == 2:
            image_patch = image_stack_average[720:880, 830:990]

        #Forth loop, bottom left corner
        else:
            image_patch = image_stack_average[720:880, 220:410]

        #Just a flag for later usage
        found_corner = False

        #Slide the Kernel over the image, which contains the corner
        for y in range(0, image_patch.shape[0] - kernel.shape[0]):       #Iterate through the rows (move kernel in Y direction)
            for x in range(0, image_patch.shape[1] - kernel.shape[1]):   #Iterate through the columns (move kernel in X direction)
                
                sliding_window = image_patch[y:(y + kernel.shape[0]), x:(x+kernel.shape[1])]
                sliding_window_org = sliding_window
                sliding_window = sliding_window * kernel
       
                window_nonzero_count = np.count_nonzero(sliding_window)

                if window_nonzero_count > (kernel_nonzero_count * threshold):

                   if i == 0:
                      #Top left corner
                      corner_points[0, 0] = 180 + (x + 5)       #X-Pos
                      corner_points[0, 1] = 80 + (y + 5)        #Y-Pos
                   elif i == 1:
                      #Top right corner
                      corner_points[1, 0] = 880 + (x + 45)      #X-Pos
                      corner_points[1, 1] = 80 + (y + 5)        #Y-Pos
                   elif i == 2:
                      #Bottom right corner
                      corner_points[2, 0] = 830 + (x + 45)      #X-Pos
                      corner_points[2, 1] = 720 + (y + 50)      #Y-Pos
                   else:
                      #Bottom left corner
                      corner_points[3, 0] = 220 + (x + 5)       #X-Pos
                      corner_points[3, 1] = 720 + (y + 45)      #Y-Pos

                   found_corner = True

                   #Break both loops....
                   break

            #.... if the corner has been detected
            if found_corner == True:
               break


        #Rotate the kernel by 90° to search for the next corner
        kernel = np.rot90(kernel, -1)


        #If the corner has not been detected, try to interpolate the location from the known corner points.
        #typically the lower left and right corners are being missed.

    while (((corner_points[0, 0] == 0) & (corner_points[0, 1] == 0)) or
          ((corner_points[1, 0] == 0) & (corner_points[1, 1] == 0)) or
          ((corner_points[2, 0] == 0) & (corner_points[2, 1] == 0)) or
          ((corner_points[3, 0] == 0) & (corner_points[3, 1] == 0))):

        #Check if no corner point has been found:
        if (not corner_points.any()) == True:
           print('No corner points have been found. Please enter the coordinate of a corner.')
           print('Note:\nTop left: 0\nTop right: 1\nBottom right: 2\nBottom left: 3')
           plt.imshow(image_stack_average, cmap='gray')
           plt.title('average image')
           plt.show()

           while True:
               try:
                  corner_number = int(input('Corner Number: '))
                  while corner_number > 3:
                     print('Corner number must be a number between 0-3.')
                     corner_number = int(input('Corner Number: '))
                  corner_points[corner_number, 0] = int(input('X-coordinate: '))
                  corner_points[corner_number, 1] = int(input('Y-coordinate: '))
               except ValueError:
                  print('Please only enter integer numbers.')
                  continue
               else:
                  break

           print('Calculating corners...')

        if (corner_points[0, 0] == 0) & (corner_points[0, 1] == 0):
           #Top left corner not found
           #Try to interpolate from the top right corner:
           if (corner_points[1, 0] != 0) & (corner_points[1, 1] != 0):
              corner_points[0, 0] = corner_points[1, 0] - 720      #X-Pos
              corner_points[0, 1] = corner_points[1, 1] - 2        #Y-Pos
           elif (corner_points[3, 0] != 0) & (corner_points[3, 1] != 0):
              #or from the bottom left corner:
              corner_points[0, 0] = corner_points[3, 0] - 46      #X-Pos
              corner_points[0, 1] = corner_points[3, 1] - 646     #Y-Pos


        if (corner_points[1, 0] == 0) & (corner_points[1, 1] == 0):
           #Top right corner not found
           #Try to interpolate from the top left corner:
           if (corner_points[0, 0] != 0) & (corner_points[0, 1] != 0):
              corner_points[1, 0] = corner_points[0, 0] + 720      #X-Pos
              corner_points[1, 1] = corner_points[0, 1] + 4        #Y-Pos
           elif (corner_points[2, 0] != 0) & (corner_points[2, 1] != 0):
              #or from the bottom right corner:
              corner_points[1, 0] = corner_points[2, 0] + 55      #X-Pos
              corner_points[1, 1] = corner_points[2, 1] - 646     #Y-Pos


        if (corner_points[2, 0] == 0) & (corner_points[2, 1] == 0):
           #Bottom right corner not found
           #Try to interpolate from the top right corner:
           if (corner_points[1, 0] != 0) & (corner_points[1, 1] != 0):
              corner_points[2, 0] = corner_points[1, 0] - 55      #X-Pos
              corner_points[2, 1] = corner_points[1, 1] + 644        #Y-Pos
           elif (corner_points[3, 0] != 0) & (corner_points[3, 1] != 0):
              #or from the bottom left corner:
              corner_points[2, 0] = corner_points[3, 0] + 614      #X-Pos
              corner_points[2, 1] = corner_points[3, 1] - 2     #Y-Pos

        
        if (corner_points[3, 0] == 0) & (corner_points[3, 1] == 0):
           #Bottom left corner not found
           #Try to interpolate from the top left corner:
           if (corner_points[0, 0] != 0) & (corner_points[0, 1] != 0):
              corner_points[3, 0] = corner_points[0, 0] + 50      #X-Pos
              corner_points[3, 1] = corner_points[0, 1] + 640        #Y-Pos
           elif (corner_points[2, 0] == 0) & (corner_points[2, 1] == 0):
              #or from the bottom right corner:
              corner_points[3, 0] = corner_points[2, 0] - 610      #X-Pos
              corner_points[3, 1] = corner_points[2, 1] + 2     #Y-Pos


    #All corner points of the powderbed has been detected.
    #Now create a list of all images in the project folder and cut out the powderbed and unwarp it.
    image_list = list(project_directory + dataframe['Image'])

    for i in range(0, len(image_list)):
        #Load the image and decode the png file
        image = tf.io.read_file(image_list[i])
        image = tf.image.decode_png(image)
        image = np.asarray(image, dtype=np.uint8)

        #Unwarp the image
        image = _unwarp(image, corner_points, powderbed_height, powderbed_width)

        #Encode the unwarped image and save the image to disk.
        image = tf.image.encode_png(image)
        tf.io.write_file(image_list[i], image)
        