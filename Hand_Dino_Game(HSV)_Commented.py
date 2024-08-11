import numpy as np
import cv2
import math
import pyautogui

# Open the camera to capture video
capture = cv2.VideoCapture(0)

# Define the window size for displaying images
window_width = 640
window_height = 480

while capture.isOpened():
    # Capture a single frame from the camera
    ret, frame = capture.read()

    # Draw a rectangle on the frame to define the region of interest (ROI) for hand data
    cv2.rectangle(frame, (100, 200), (300, 400), (0, 255, 0), 0)
    # Extract the portion of the frame within the rectangle
    crop_image = frame[200:400, 100:300]

    # Apply Gaussian blur to the cropped image to smooth it and reduce noise
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Convert the blurred image from BGR (Blue-Green-Red) to HSV (Hue-Saturation-Value) color space
    # This is often more effective for color-based segmentation
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary mask based on the HSV color range to isolate skin colors
    # This mask will be white for colors in the specified range and black otherwise
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Define a kernel for morphological transformations
    # A kernel is a small matrix used for image processing operations
    kernel = np.ones((3, 3))

    # Apply dilation to the binary mask to expand white regions (skin color) in the mask
    # Dilation helps to fill small holes and gaps in the detected skin areas
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    # Apply erosion to the dilated mask to shrink the white regions and remove noise
    # Erosion helps to remove small noise points and smooth the edges of the detected skin areas
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian blur to the eroded mask to further smooth it
    # This helps in reducing noise and improving the quality of the binary image
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    # Apply binary thresholding to the filtered image
    # Pixels with intensity greater than 170 are set to 255 (white), otherwise set to 0 (black)
    ret, thresh = cv2.threshold(filtered, 170, 255, 0)

    # Find contours in the thresholded image
    # Contours are the boundaries of white regions in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find the contour with the maximum area, which is likely the hand
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create a bounding rectangle around the largest contour
        # This rectangle helps to enclose the detected hand area
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding rectangle on the cropped image
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Compute the convex hull of the contour
        # The convex hull is the smallest convex shape that encloses the contour
        hull = cv2.convexHull(contour)

        # Create a blank image for drawing contours
        drawing = np.zeros(crop_image.shape, np.uint8)
        # Draw the contour on the blank image
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        # Draw the convex hull on the blank image
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects of the contour using the convex hull
        # Convexity defects are the gaps or indentations in the convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Initialize a counter for the number of defects
        count_defects = 0

        # Process each convexity defect
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Calculate the lengths of the sides of the triangle formed by the defect points
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (end[1] - far[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

            # Calculate the angle of the defect using the cosine rule
            # This angle is used to determine whether the defect corresponds to a finger tip
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # If the angle is less than or equal to 90 degrees, it indicates a finger tip
            if angle <= 90:
                count_defects += 1
                # Draw a circle at the farthest point of the defect
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            # Draw lines between the start, end, and far points of the defect
            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # If there are 4 or more defects, it indicates a gesture (e.g., open hand)
        # Simulate a spacebar press and display a "JUMP" message on the frame
        if count_defects >= 4:
            pyautogui.press('space')
            cv2.putText(frame, "JUMP", (115, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

    except:
        # If an exception occurs (e.g., no contours found), continue to the next frame
        pass

    # Display the processed images in separate windows with the specified size
    cv2.imshow("Cropped Image", crop_image)
    cv2.resizeWindow("Cropped Image", window_width, window_height)

    cv2.imshow("Blurred Image", blur)
    cv2.resizeWindow("Blurred Image", window_width, window_height)

    cv2.imshow("HSV Image", hsv)
    cv2.resizeWindow("HSV Image", window_width, window_height)

    cv2.imshow("Binary Mask", mask2)
    cv2.resizeWindow("Binary Mask", window_width, window_height)

    cv2.imshow("Erosion Image", erosion)
    cv2.resizeWindow("Erosion Image", window_width, window_height)

    cv2.imshow("Thresholded Image", thresh)
    cv2.resizeWindow("Thresholded Image", window_width, window_height)

    cv2.imshow("Contours and Hull", drawing)
    cv2.resizeWindow("Contours and Hull", window_width, window_height)

    cv2.imshow("Gesture", frame)
    cv2.resizeWindow("Gesture", window_width, window_height)

    # Exit the loop and close all windows if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()
