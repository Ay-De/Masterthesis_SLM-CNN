import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs, 2: filter out warning logs, 3: filter out error logs
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from time import sleep
import glob

plt.rcParams.update({'font.size': 16})

#Fix for the Tensorflow error: Failed to get convolution algorithm.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('')


#Split the complete slice layer into not overlapping patches of 128x128 pixels,
#which will be used for classification
def split_image(source_image, window_size, window_overlap, source_dim):

    patch_data = []

    #Check if there is a 50% overlap between patches or no overlap at all.
    #Required for the patch behavior on the right and lower sides
    if window_overlap == window_size:
       last_step = 0

    else:
       last_step = window_overlap

    for ypos in range(0, (source_dim - last_step), window_overlap):
            for xpos in range(0, (source_dim - last_step), window_overlap):
                
                #Initialize the arrays to contain the patch
                sliding_window = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)
                image_patch = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)

                #Copy each image data channel from the source image to the patch
                for channel in range(0, 3):
                    sliding_window = source_image[ypos : (ypos + window_size), xpos : (xpos + window_size), channel]
                    image_patch[:, :, channel] = sliding_window

                #And store the patch in a list
                patch_data.append(image_patch)
                
    return np.asarray(patch_data)


#Function to load the dataset
#Example input:
#'D:\\3D_Druck_Datensatz_Backup_20201203\\Datensatz_Pulverbett_Extrahiert\\Test_Data\\Daten_2018\\2018-06-12_152613\\'
def load_project_imagepaths_labels(project_folder):

    #Load the csv file containing image filenames and labels
    dataframe = pd.read_csv(project_folder + 'logs\\image_label.csv', sep=';')

    #Drop column 'Drop_Images'
    dataframe.drop(columns=['Drop_Image'], axis=1, inplace=True)

    dataframe['Image'] = project_folder + 'logs\\exposures\\' + dataframe['Image'].astype(str)

    return list(dataframe['Image']), list(dataframe['Label_Binary'])


#This function will create an overlay image. Each Patch will have either the color
#Green (Pulver), blue (Bauteil/object) or red (Fehler/Error)
#Inputs:
#   predictions_logits: Predicted logits for each patch from the CNN (0: Pulver, 1: Bauteil, 2: Fehler)
#   predictions_logits has 36 elements. (patch size 128, source dim 768x768, no overlap -> 768/128=6 -> 6x6=36 Patches)
def stitch_image(predictions_logits, window_size, window_overlap, source_dim):

    #Initialize an empty array with the shape of the final image (the extracted powderbed)
    image = np.zeros(shape=(source_dim, source_dim, 3), dtype=np.uint8)

    #Check if there is a 50% overlap between patches or no overlap at all.
    if window_overlap == window_size:
       last_step = 0

    else:
       last_step = window_overlap

    #Counter to determine patch number/patch location
    i = 0

    for ypos in range(0, (source_dim - last_step), window_overlap):
            for xpos in range(0, (source_dim - last_step), window_overlap):

                #initialize a patch
                color = np.zeros(shape=(window_size, window_size, 3), dtype=np.uint8)

                #Set the patch color according to the CNN Label
                if predictions_logits[i] == 0:
                   color[...,0:3]=[0,128,0] #Green for powder
                elif predictions_logits[i] == 1:
                   color[...,0:3]=[0,0,255] #blue for object
                else:
                   color[...,0:3]=[255,0,0] #red for error

                #Fill the empty powderbed array from above with the patches and their colors
                for channel in range(0, 3):
                    image[ypos : (ypos + window_size), xpos : (xpos + window_size), channel] = color[:,:,channel]

                #increase patch number by 1 (move to the next patch)
                i = i + 1

    return image


#Function to load and decode a PNG image
def load_image(img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3, dtype=np.uint8)
    
    return np.asarray(image)


#Normalize the image patches before making predictions on them
def normalize(image_patches):
    
    image_patches = tf.image.convert_image_dtype(image_patches, tf.float32)
    image_patches = tf.image.per_image_standardization(image_patches)

    return image_patches


def main():
    #########################################################################
    #Function to visualize the powderbed in real time. The image overlay and the layer label are updated
    #after each layer. Each Layer will get a binary Label (0 Good Powderbed, 1 Error in the Powderbed detected)
    #########################################################################

    #Path to the CNN Model
    model_path = 'D:\\Masterarbeit\\Daten\\Modell\\'
    #Path to the Project folder containing the images
    project_folder = 'D:\\Masterarbeit\\Daten\\Datensatz_Komplett\\Datensatz_Pulverbett_Extrahiert\\Test_Data\\Daten_2018\\2018-06-12_152613\\'


    patch_size = 128
    patch_overlap = 128 #if patch_size == patch_overlap -> No overlap between two patches
    source_image_dim = 768 #Powderbed dimension
    
    #Load the cnn model
    cnn_model = tf.keras.models.load_model(model_path)

    #Get the list of all image filenames and their true slice labels.
    #Note: Slice Label is binary and applies to the whole Slice Layer. 0 -> Good Powderbed. 1 -> There is an error in the Layer
    project_images, true_slice_labels = load_project_imagepaths_labels(project_folder)
    
    #Needed for later to display the plot
    first_loop = True

    fig, ax = plt.subplots()
    plt.axis('off')
    plt.title('Pulver: Grün | Bauteil: Blau | Fehler: Rot')
    img_data_object = ax.imshow(np.zeros(shape=(source_image_dim, source_image_dim, 3), dtype=np.uint8), interpolation='nearest')
    fig.canvas.draw()

    #########################################################################
    #NOTE: FOR IN PRODUCTION USAGE
    #########################################################################
    #The following variables should be moved into the main function of the code and initialized upon starting the print of an object.
    #CNN output: array with 3 probabilities that a patch is part of a class.
    #Example: Patch 0: [0.10 0.70 0.20], Patch 1: [0.99 0.00 0.00], ...
    #Each Powderbed consists of 36 not overlapping patches (kind of like a chessboard)
    #The historic development of each patch is stored here.
    patchwise_pulver_raw = [[] for _ in range(36)]
    patchwise_bauteil_raw = [[] for _ in range(36)]
    patchwise_fehler_raw = [[] for _ in range(36)]

    #Specifies how many layers back in time are averaged over.
    window_length = 5

    #The historic average over each patch is stored in this variable.
    #The first 5 zeros are needed because the averaging looks 5 Layers back in time.
    #first average happens with the fifth layer
    patchwise_avg_pulver = [[0,0,0,0,0] for _ in range(36)]
    patchwise_avg_bauteil = [[0,0,0,0,0] for _ in range(36)]
    patchwise_avg_fehler = [[0,0,0,0,0] for _ in range(36)]

    #Variables to store the binary label for the entire powderbed (layer) from the raw CNN and the averaged CNN values
    #Note the 5 zero values because the averaging is looking 5 layers back in time
    predicted_slice_labels_cnn = []
    predicted_slice_labels_avg = [0,0,0,0,0]

    #########################################################################
    #NOTE: FOR IN PRODUCTION USAGE
    #########################################################################
    #The following for loop has to be removed and the code inside of it has to be
    #cyclically run in parallel to the printing process. Once a powderbed Layer
    #has been completed and a picture was taken, the code inside this for loop
    #has to be executed to make a prediction.
    #
    #Note:
    #Don't forget the powderbed preprocessing (should be included here after loading the raw powderbed image)
    #call either powderbed_detector function or direct _unwarp if the absolute pixel positions of the powderbed are known.
    #########################################################################
    for i in range(len(project_images)):
        #Load the image file (specify the image path to a slice)
        image_data = load_image(project_images[i])
        #split the (extracted) powderbed image into not overlapping patches and stack them into a list
        patch_array = split_image(image_data, patch_size, patch_overlap, source_image_dim)
        #Normalize the patches  for prediction (subtract mean and divide by standard deviation)
        patches_normalized = normalize(patch_array)
        #predict all patches from one powderbed (Layer)
        y_predicted_raw = cnn_model.predict(patches_normalized, verbose=0)
        #Get the CNN Logits with argmax for each patch
        y_predicted_logits = np.argmax(y_predicted_raw, axis=-1)

        #create a list of the patch logits for the current powderbed layer
        cnn_logits = list(y_predicted_logits)
        #The corner points (patch number 0, 5, 30 and 35) are ignored for the binary layer label (0 good or 1 bad)
        #There are usually no objects printed and the corners are a weakpoint of the CNN
        cnn_logits.pop(35)
        cnn_logits.pop(30)
        cnn_logits.pop(5)
        cnn_logits.pop(0)

        #If any of the classified patches has the label '2' (Error/Fehler), append a binary label 1 for Error.
        if any(p == 2 for p in cnn_logits):
            predicted_slice_labels_cnn.append(1)
        #Else Append for this current slice a 0 (no error)
        else:
            predicted_slice_labels_cnn.append(0)

        #Append the raw CNN predictions of each class for each patch. 
        for patchnumber in range(0,36):
            patchwise_pulver_raw[patchnumber].append(y_predicted_raw[patchnumber][0])
            patchwise_bauteil_raw[patchnumber].append(y_predicted_raw[patchnumber][1])
            patchwise_fehler_raw[patchnumber].append(y_predicted_raw[patchnumber][2])

        #if the current image (=slice) is over 5 -> calculate the weighted average for each patch
        if i >= window_length:

            for patchnumber in range(0,36):
                weighted_average_pulver = np.average(patchwise_pulver_raw[patchnumber][-window_length:], weights=[0.05, 0.05, 0.2, 0.3, 0.4])
                patchwise_avg_pulver[patchnumber].append(weighted_average_pulver)
                
                weighted_average_bauteil = np.average(patchwise_bauteil_raw[patchnumber][-window_length:], weights=[0.05, 0.05, 0.2, 0.3, 0.4])
                patchwise_avg_bauteil[patchnumber].append(weighted_average_bauteil)
                
                weighted_average_fehler = np.average(patchwise_fehler_raw[patchnumber][-window_length:], weights=[0.05, 0.05, 0.2, 0.3, 0.4])
                patchwise_avg_fehler[patchnumber].append(weighted_average_fehler)
            
            #Rebuild the CNN output with the weighted average values to be able to apply the argmax function on them
            y_weighted_raw = [[lst[i] for lst in patchwise_avg_pulver], [lst[i] for lst in patchwise_avg_bauteil], [lst[i] for lst in patchwise_avg_fehler]]
            
            #Get the argmax for each patch
            y_weighted_logits = np.argmax(y_weighted_raw, axis=0)

            #Get from all patches in the current layer the values of the weighted average error class.
            #The Probability that each patch is part of the class Error/Fehler
            slice_label = [lst[i] for lst in patchwise_avg_fehler]

            #Drop again the 4 Corner points
            slice_label.pop(35)
            slice_label.pop(30)
            slice_label.pop(5)
            slice_label.pop(0)

            #Check if any of the remaining patches is for more than 70% part of the class Error/Fehler
            if any(p > 0.7 for p in slice_label):
               predicted_slice_labels_avg.append(1)
            #If none of the patches in the current layer is for at least 70% part of the class Fehler -> append 0 (Good layer)
            else:
               predicted_slice_labels_avg.append(0)

        #Create the chessboard like image with each patch containing the color of their label.
        #If we are in the first 5 slices (where no weighted average is calculated) use the raw CNN values
        stitched_image = stitch_image((y_weighted_logits if i >= window_length else y_predicted_logits), patch_size, patch_overlap, source_image_dim)
        #overlay the original image with the color map to show how each patch was classified
        overlayed_images = cv2.addWeighted(image_data, 0.7, stitched_image, 0.2, 0)
        
        plt.title(('Ok' if predicted_slice_labels_avg[i] == 0 else 'Fehler'), backgroundcolor=('green' if predicted_slice_labels_avg[i] == 0 else 'red'))

        #Some Code to update the Matplotlib window
        img_data_object.set_data(overlayed_images)
        fig.canvas.draw()
        plt.pause(0.001)
        if first_loop == True:
            first_loop = False
            plt.ion()
            plt.show()
        sleep(0.2) 

    plt.ioff()

    #This Code is kind of a report
    #The following code will display the True binary labels for each slice of the printed object...
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(true_slice_labels, linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 1], ['Fehlerfrei', 'Fehler'])
    #plt.legend(loc='upper right', prop={"size":11})
    plt.title('Tatsächliches Schichtlabel')

    #The binary label if only the raw CNN output (with argmax) was used to determine the total slice label...
    plt.subplot(3, 1, 2)
    plt.plot(predicted_slice_labels_cnn, linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 1], ['Fehlerfrei', 'Fehler'])
    #plt.legend(loc='upper right', prop={"size":11})
    plt.title('Vorhergesagtes Schichtlabel CNN')
    #plt.xlabel('Schichtnummer', fontweight='bold', fontsize=20, labelpad=10)
    plt.ylabel('Klassenzugehörigkeit', fontweight='bold', fontsize=20, labelpad=10)

    #And the binary slice label if the weighted average with a threshold of 70% is used to determine the slice label
    #Note: This is method (weighted average + threshold) is what is seen in the matplot video.
    plt.subplot(3, 1, 3)
    plt.plot(predicted_slice_labels_avg, linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 1], ['Fehlerfrei', 'Fehler'])
   # plt.legend(loc='upper right', prop={"size":11})
    plt.title('Vorhergesagtes Schichtlabel mit Schwellwert')
    plt.xlabel('Schichtnummer', fontweight='bold', fontsize=20, labelpad=10)
    #plt.ylabel('Klassenzugehörigkeit', fontweight='bold', fontsize=20, labelpad=10)
    plt.tight_layout()



    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(patchwise_pulver_raw[8], label='CNN Ausgabe', linewidth=2)
    plt.plot(patchwise_avg_pulver[8], label='Gewichteter Mittelwert', linestyle='--', linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 0.5, 1])
    plt.legend(loc='upper right', prop={"size":11})
    plt.title('Pulver')

    plt.subplot(3, 1, 2)
    plt.plot(patchwise_bauteil_raw[8], label='CNN Ausgabe', linewidth=2)
    plt.plot(patchwise_avg_bauteil[8], label='Gewichteter Mittelwert', linestyle='--', linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 0.5, 1])
    plt.legend(loc='upper right', prop={"size":11})
    plt.title('Bauteil')
    #plt.xlabel('Schichtnummer', fontweight='bold', fontsize=20, labelpad=10)
    plt.ylabel('Klassenzugehörigkeit', fontweight='bold', fontsize=20, labelpad=10)

    #And the binary slice label if the weighted average with a threshold of 70% is used to determine the slice label
    #Note: This is method (weighted average + threshold) is what is seen in the matplot video.
    plt.subplot(3, 1, 3)
    plt.plot(patchwise_fehler_raw[8], label='CNN Ausgabe', linewidth=2)
    plt.plot(patchwise_avg_fehler[8], label='Gewichteter Mittelwert', linestyle='--', linewidth=2)
    plt.ylim([-0.02, 1.02])
    plt.yticks([0, 0.5, 1])
    plt.legend(loc='upper right', prop={"size":11})
    plt.title('Fehler')
    plt.xlabel('Schichtnummer', fontweight='bold', fontsize=20, labelpad=10)
    #plt.ylabel('Klassenzugehörigkeit', fontweight='bold', fontsize=20, labelpad=10)
    plt.tight_layout()


    plt.show()








if __name__ == '__main__':
    main()