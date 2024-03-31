import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('seismic7.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 170, 100])
upper_blue = np.array([140, 255, 255])

# lower_blue = np.array([0, 100, 100])
# upper_blue = np.array([160, 100, 100])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
filtered_img = cv2.bitwise_and(img, img, mask=mask)


gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
inverted = 255 - thresh

# Temukan kontur dalam citra mask
contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Inisialisasi citra baru untuk menampilkan hasil kontur yang difilter
filtered_contours_img = np.zeros_like(inverted)
# Iterasi melalui setiap kontur
for contour in contours:
    # Hitung luas kontur
    contour_area = cv2.contourArea(contour)
    # Jika luas kontur memenuhi syarat, gambar kontur pada citra baru
    if contour_area > 2.5:

        epsilon = 0.01* cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(filtered_contours_img, [approx], -1, 255, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
     
        # # Calculate the minimum bounding rectangle of the contour
        # rect = cv2.minAreaRect(contour)
        # # Get the center and orientation angle of the rectangle
        # center, size, angle = rect
        # # Get the endpoints of the middle line
        # p1 = (int(center[0]), int(center[1]))
        # p2 = (int(center[0] + np.cos(np.radians(angle)) * size[0] / 2),
        #     int(center[1] + np.sin(np.radians(angle)) * size[0] / 2))
        # # Draw the middle line on the filtered_contours_img
        # cv2.line(filtered_contours_img, p1, p2, (255, 255, 255), 2)
peler = cv2.bitwise_not(filtered_contours_img)
kernel = np.ones((4,4),np.uint8)
erosion = cv2.erode(filtered_contours_img,kernel,iterations = 1)
closing = cv2.morphologyEx(peler, cv2.MORPH_CLOSE, kernel)


# cv2.imshow('Original', img)
# cv2.imshow('Filtered Image', filtered_img)
# cv2.imshow('Inverted Image', inverted)
# cv2.imshow('filtered_contours_imgq',filtered_contours_img)
# cv2.imshow('EROSION',erosion)
# cv2.imshow('closing',closing)
cv2.imshow('Result',peler)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Ubah format citra dari BGR ke RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtered_img_rgb = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)
inverted_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
filtered_contours_img_rgb = cv2.cvtColor(filtered_contours_img, cv2.COLOR_GRAY2RGB)
erosion_rgb = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
closing_rgb = cv2.cvtColor(closing, cv2.COLOR_GRAY2RGB)
peler_rgb = cv2.cvtColor(peler, cv2.COLOR_GRAY2RGB)

# Tampilkan gambar menggunakan matplotlib
plt.figure(figsize=(10, 8))

plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title('Original')

plt.subplot(3, 3, 2)
plt.imshow(filtered_img_rgb)
plt.title('Filtered Image')

plt.subplot(3, 3, 3)
plt.imshow(inverted_rgb)
plt.title('Inverted Image')

plt.subplot(3, 3, 4)
plt.imshow(filtered_contours_img_rgb)
plt.title('Filtered Contours Image')

plt.subplot(3, 3, 5)
plt.imshow(erosion_rgb)
plt.title('Erosion')

plt.subplot(3, 3, 6)
plt.imshow(closing_rgb)
plt.title('Closing')

plt.subplot(3, 3, 7)
plt.imshow(peler_rgb)
plt.title('Seismic Internal Character')

plt.tight_layout()
plt.show()
