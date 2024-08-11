# CVPresentation

HandDino is a Python application that allows you to control Chrome's Dinosaur game using hand movements. Utilizing OpenCV, this program processes hand gestures to simulate keyboard inputs, enabling you to play the game without touching your keyboard.

Packages Required
-OpenCV (cv2): For computer vision tasks, such as image processing and gesture detection.

-numpy: For numerical operations, especially with arrays.

-math: For mathematical calculations used in gesture analysis.

-pyautogui: For simulating keyboard inputs based on detected gestures.

Execution

Setup: Ensure you have the required packages installed. 

Run the Program: Execute the Python script. Make sure to have a webcam connected to your computer.

Start the Game:
Open Chrome and navigate to chrome://dino.
Keep your hand within the rectangle drawn by the program on the video feed.

Play:
Move your hand within the defined area to make the dinosaur jump in the game.

Program Workflow
Capture Frame:

The program captures a frame from the webcam in real-time.
Define Region of Interest (ROI):

A rectangle is drawn on the frame to focus on the area where hand movements will be detected.
The area inside this rectangle is extracted for further processing.
Gaussian Blur:

Applied to the extracted region to smooth the image and reduce noise.
Convert to HSV Color Space:

The image is converted from BGR (Blue-Green-Red) to HSV (Hue-Saturation-Value) color space. This conversion enhances color-based segmentation.
Create Binary Mask:

A binary mask is created to highlight skin tones in the HSV image. This mask isolates the skin color, making it easier to detect hand movements.
Morphological Transformations:

Dilation: Expands the white regions in the mask to fill small gaps.
Erosion: Shrinks the white regions to remove noise and smooth edges.
Thresholding:

The mask is further processed to create a binary image where significant features are highlighted.
Contour Detection:

Contours are found in the binary image. The largest contour, assumed to be the hand, is selected for further analysis.
Contour Analysis:

A bounding rectangle is drawn around the largest contour to highlight the hand's position.
The convex hull (the smallest convex shape enclosing the contour) is computed and drawn.
Convexity Defects Detection:

Convexity defects are identified to detect indentations in the hand's contour, which are used to determine finger tips.
Gesture Recognition:

Using the convexity defects, the program calculates angles to identify finger tips.
If a sufficient number of finger tips are detected (indicative of a specific gesture), a corresponding keyboard action is simulated (e.g., pressing the spacebar to make the dinosaur jump).
Display Results:

Various stages of image processing are displayed in separate windows for debugging and analysis.
Tests and Controls
Adjust HSV Values: Modify the HSV range values in the cv2.inRange function to match your lighting conditions for better skin color detection.
Test with Different Lighting: Ensure that the program works effectively under different lighting conditions by testing and adjusting parameters as needed.
