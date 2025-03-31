import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('picture.png')

# Get image dimensions
# Now its a 3D matrix ig so, ch is it depth (RGB hence it should be 3)
ht, wd, ch = image.shape
print(f"Height: {ht}, Width: {wd}, Channels: {ch}")

newimage = np.copy(image)

# Reflect the upper part into the lower part for each color channel
for c in range(3):  # Doing the samething for B, G, R channels
    for i in range(ht // 2):
        for j in range(wd):
            newimage[ht - 1 - i, j, c] = image[i, j, c]

    for j in range(wd // 2):
        for i in range(ht):
            newimage[i, wd - 1 - j, c] = newimage[i, j, c]

# Convert BGR to RGB for displaying with Matplotlib
newimage_rgb = cv.cvtColor(newimage, cv.COLOR_BGR2RGB)

# Display the result
plt.imshow(newimage_rgb)
plt.title('Reflected Color Image')
plt.axis('off')
plt.show()
