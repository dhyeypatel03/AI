# import cv2
# import os
# from cv2 import dnn_superres

# # Create an SR object
# sr = dnn_superres.DnnSuperResImpl_create()

# # Read the desired model
# path = 'C:/EDSR_x4.pb'
# sr.readModel(path)

# # Set the desired model and scale to get correct pre- and post-processing
# sr.setModel("edsr", 3)

# # Input and output folder paths
# input_folder = 'C:/ANPR3/Detection_Images'
# output_folder = 'C/ANPR3/SHARP_IMAGE_OUTPUT'

# # Check if output folder exists, if not, create it
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# # Get a list of all image files in the input folder
# image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# # Process each image in the input folder
# for image_file in image_files:
#     # Read image
#     image_path = os.path.join(input_folder, image_file)
#     image = cv2.imread(image_path)

#     # Upscale the image
#     result = sr.upsample(image)

#     # Save the upscaled image to the output folder
#     output_path = os.path.join(output_folder, f"upscaled_{image_file}")
#     cv2.imwrite(output_path, result)

# print("Image upscaling completed.")

import cv2
import os
from cv2 import dnn_superres


# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "EDSR_x3.pb"
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 3)

# Input and output folder paths
input_folder = 'C:/ANPR3/Detection_Images'
output_folder = 'C/ANPR3/SHARP_IMAGE_OUTPUT'

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Process each image in the input folder
for image_file in image_files:
    # Read image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Upscale the image
    result = sr.upsample(image)

    # Save the upscaled image to the output folder
    output_path = os.path.join(output_folder, f"upscaled_{image_file}")
    cv2.imwrite(output_path, result)

print("Image upscaling completed.")
