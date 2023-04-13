import cv2
import cv2.aruco as aruco

# Load the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)

# Create the detector parameters
parameters = aruco.DetectorParameters_create()


# Load the image
img = cv2.imread('ar_marker.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the markers
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Draw a border around the markers
img_with_border = aruco.drawDetectedMarkers(img, corners, borderColor = (255,0,0))

# Display the image with borders
cv2.imshow('Markers with Borders', img_with_border)
cv2.waitKey(0)
cv2.destroyAllWindows()
