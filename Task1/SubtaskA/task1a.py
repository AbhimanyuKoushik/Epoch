import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Cv means Computer vision, interesting

# imread means image read, It reads the image into image (variable), nice..
image = cv.imread('picture.png')
# Apparently, cv stores the matrix in BGR (weird) and not in RGB format
# So for gray scale we have to convert from BGR to Gray scale
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# image.shape outputs the height and width of image in pixels
ht, wd = image.shape
# Just checking
print(f"Height: {ht}, Width: {wd}")

# Create a copy of the image so that we don't disturb the original image
newimage = np.copy(image)

# According to the task we have to reflect whatever is there on above half on lower half
# And then whatever is there on left half to right half
for i in range(ht // 2):
    # i ranges from 0 to h/2 (only one half should be reflected hence h/2)
    for j in range(wd):
        # j goes from 0 to wd (all the image should get reflected, not just some part along x-axis)
        newimage[ht - 1 - i, j] = image[i, j]
        # Whatever is the value of pixel in the above half, it gets copied to the below half
# Ig what is going on, after we changes to gray scale, each pixel has a value from 0 to 255
# 0 meaning white and 255 meaning black and in between gray, so what ever the value of the colour it gets stored in new image 

# Do a similar thing for width except make sure that now you are working with newimage
# I made a mistake which gave weird output, so just in case
for j in range(wd//2):
    for i in range(ht):
        # if it was newimage[i, wd - 1 - j] = image[i, j], the bottom right half is left untouched (since image is untouched)
        newimage[i, wd - 1 - j] = newimage[i, j]

# Display the result
plt.imshow(newimage, cmap='gray')
plt.title('Reflected Color Image')
plt.axis('off')
plt.show()

