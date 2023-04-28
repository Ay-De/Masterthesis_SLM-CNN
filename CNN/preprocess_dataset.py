import tensorflow as tf
from modules.preprocessing import load_image_paths_labels, powderbed_detector
import glob

#Fix for the Tensorflow error: Failed to get convolution algorithm.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('')

def main():
    #NOTE:
    #Pictures will be overwritten!
    #Target dimensions of the extracted powderbed
    IMG_HEIGHT = 768
    IMG_WIDTH = 768

    dataset_directory = 'F:\\Daten\\Demo\\Original\\'

    #Create a list of all printed projects
    printed_projects_train = glob.glob(dataset_directory + 'Training_Data\\' + '**/**/')
    printed_projects_val = glob.glob(dataset_directory + 'Validation_Data\\' + '**/**/')
    printed_projects_test = glob.glob(dataset_directory + 'Test_Data\\' + '**/**/')

    printed_projects = printed_projects_train + printed_projects_val + printed_projects_test

    #Detect the powderbed in each image and unwarp it before storing it back to disk
    for i in range(0, len(printed_projects)):
        if len(printed_projects) > 1:
            progress = (i/(len(printed_projects) - 1))*100
            print('Image preprocessing...', str(round(progress, 2)), '%')
        else:
            print('Image preprocessing...')
        powderbed_detector(printed_projects[i], IMG_HEIGHT, IMG_WIDTH)
        print('Image preprocessing... Done')



if __name__ == "__main__":
    main()