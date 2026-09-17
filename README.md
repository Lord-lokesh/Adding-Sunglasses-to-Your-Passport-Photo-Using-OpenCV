# Adding-Sunglasses-to-Your-Passport-Photo-Using-OpenCV
Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

# Features:

Detects the face in an image.

Places a stylish sunglass overlay perfectly on the face.

Works seamlessly with individual passport-size photos.

Customizable for different sunglasses styles or photo types.

# Technologies Used:

Python

OpenCV for image processing

Numpy for array manipulations

# How to Use:

Clone this repository.

Add your passport-sized photo to the images folder.

Run the script to see your "cool" transformation!

#Applications:

Learning basic image processing techniques.

Adding flair to your photos for fun.

Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

# Program:

NAME: Lokesh M

REGISTER NUMBER: 212224230142

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the Face Image
faceImage = cv2.imread('web lok.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")



faceImage.shape



#resized_faceImage.shape
faceImage.shape


# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sun.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")



# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))


# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]



# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');



# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[135:185,110:300]=glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])




# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[135:185,110:300]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1-  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR,glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")


# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[135:185,110:300]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");



```
<img width="403" height="480" alt="image" src="https://github.com/user-attachments/assets/f89994ad-496f-4a98-8826-5bc75f45c4c1" />


<img width="722" height="266" alt="image" src="https://github.com/user-attachments/assets/4f70925a-ce21-4871-b2c9-ec8c9b7a7ded" />


<img width="1252" height="212" alt="image" src="https://github.com/user-attachments/assets/ed06ce24-7cb9-404f-ab38-cf7f0b2c1b53" />

<img width="482" height="466" alt="Screenshot 2026-09-17 103604" src="https://github.com/user-attachments/assets/110b70d4-9736-4e39-aac6-c59de98b4d27" />

<img width="1235" height="816" alt="image" src="https://github.com/user-attachments/assets/e26884e0-b47a-432a-9b1e-769edfb4ce8e" />




