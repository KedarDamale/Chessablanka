import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def click_event(event, x, y, flags, param):
    """
    Callback function to capture mouse clicks on the image.
    
    Args:
        event: OpenCV mouse event type
        x, y: Coordinates of mouse pointer
        flags: Additional flags passed by OpenCV
        param: User parameters (points list and display image)
    """
    clicked_points, image_display = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point if less than 4 points are selected
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            # Draw circle at clicked point
            cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
            # Draw lines between points
            if len(clicked_points) > 1:
                cv2.line(image_display, clicked_points[-2], clicked_points[-1], (0, 255, 0), 2)
            # If we have 4 points, close the polygon
            if len(clicked_points) == 4:
                cv2.line(image_display, clicked_points[-1], clicked_points[0], (0, 255, 0), 2)
            # Display updated image
            cv2.imshow("Select 4 corners", image_display)
    
    # Show hovering coordinates as a guide
    elif event == cv2.EVENT_MOUSEMOVE:
        img_copy = image_display.copy()
        cv2.putText(img_copy, f"({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Select 4 corners", img_copy)

def get_user_points_and_crop(image_path, output_size=(800, 800)):
    """
    Allows the user to select 4 points on an image and performs a perspective transform.

    Args:
        image_path (str): Path to the input image.
        output_size (tuple): Desired size of the output warped image (width, height).

    Returns:
        tuple: A tuple containing the selected points and the warped image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")
    
    # Create a working copy for display
    display_img = image.copy()
    clicked_points = []
    
    # Set up the window with instructions
    window_name = "Select 4 corners"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, image.shape[1]), min(900, image.shape[0]))
    
    # Instructions overlay
    instructions = [
        "Click to select 4 corners in clockwise order starting from top-left",
        "Press 'r' to reset points",
        "Press 'c' to continue when 4 points are selected",
        "Press 'q' to quit"
    ]
    
    # Add instructions to the image
    for i, text in enumerate(instructions):
        cv2.putText(display_img, text, (10, 30 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.setMouseCallback(window_name, click_event, (clicked_points, display_img))
    
    while True:
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Reset points if 'r' is pressed
        if key == ord('r'):
            clicked_points.clear()
            display_img = image.copy()
            # Re-add instructions
            for i, text in enumerate(instructions):
                cv2.putText(display_img, text, (10, 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Continue if 'c' is pressed and 4 points are selected
        elif key == ord('c') and len(clicked_points) == 4:
            break
            
        # Quit if 'q' is pressed
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    if len(clicked_points) != 4:
        raise ValueError("Exactly 4 points must be selected.")
    
    # Order points in [top-left, top-right, bottom-right, bottom-left] order
    # This ensures consistent perspective transformation
    ordered_points = order_points(np.array(clicked_points))
    
    # Define the destination points (output rectangle)
    dst = np.array([
        [0, 0],                          # top-left
        [output_size[0], 0],             # top-right
        [output_size[0], output_size[1]], # bottom-right
        [0, output_size[1]]              # bottom-left
    ], dtype='float32')
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_points, dst)
    
    # Apply the transformation
    warped = cv2.warpPerspective(image, matrix, output_size)
    
    return ordered_points.tolist(), warped

def order_points(pts):
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: numpy array of 4 points
        
    Returns:
        numpy array of ordered points
    """
    # Initialize ordered points array
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum of coordinates gives the top-left and bottom-right points
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right: largest sum
    
    # Difference of coordinates gives the top-right and bottom-left points
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest difference
    
    return rect

def preview_warped_image(original, warped, points):
    """
    Display original image with selected points and the resulting warped image
    
    Args:
        original: Original image
        warped: Perspective-transformed image
        points: Selected corner points
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image with selected points
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    ax1.imshow(original_rgb)
    ax1.set_title("Original Image with Selected Points")
    
    # Plot points and connect them to form a quadrilateral
    points = np.array(points)
    for i in range(4):
        ax1.plot(points[i][0], points[i][1], 'ro')  # Plot points
        # Connect points with lines
        ax1.plot([points[i][0], points[(i+1)%4][0]], 
                 [points[i][1], points[(i+1)%4][1]], 'g-')
    
    # Display warped image
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    ax2.imshow(warped_rgb)
    ax2.set_title("Perspective Transformed Image")
    
    plt.tight_layout()
    plt.show()

