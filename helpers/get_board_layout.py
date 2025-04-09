import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

def click_and_drag_points(event, x, y, flags, param):
    state = param
    image_display = state["display_img"]
    orig_image = state["orig_img"]
    
    if state["mode"] == "drawing":
        # Original drawing functionality
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["top_left"] = (x, y)
            state["points"] = []
            
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            # Create a copy of original image for display
            img_copy = orig_image.copy()
            
            # Draw rectangle from start point to current position
            cv2.rectangle(img_copy, state["top_left"], (x, y), (0, 255, 0), 2)
            
            # Calculate all four corners
            top_left = state["top_left"]
            top_right = (x, state["top_left"][1])
            bottom_right = (x, y)
            bottom_left = (state["top_left"][0], y)
            
            # Draw corner points
            corners = [top_left, top_right, bottom_right, bottom_left]
            for i, point in enumerate(corners):
                cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                label = ["TL", "TR", "BR", "BL"][i]
                cv2.putText(img_copy, label, (point[0] + 10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            add_instructions(img_copy, state["instructions"])
            cv2.imshow(state["window_name"], img_copy)
            
        elif event == cv2.EVENT_LBUTTONUP:
            if state["drawing"]:
                state["drawing"] = False
                state["points"] = [
                    state["top_left"],
                    (x, state["top_left"][1]),
                    (x, y),
                    (state["top_left"][0], y)
                ]
                state["mode"] = "adjusting"  # Switch to adjustment mode
                state["display_img"] = update_display_image(orig_image.copy(), state["points"], None, None)
                add_instructions(state["display_img"], state["instructions"])
                cv2.imshow(state["window_name"], state["display_img"])
    
    elif state["mode"] == "adjusting":
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked near any corner point
            for i, point in enumerate(state["points"]):
                if math.sqrt((x - point[0])**2 + (y - point[1])**2) < state["point_drag_radius"]:
                    state["dragging_point"] = i
                    break
                    
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging_point"] is not None:
            # Update point position
            state["points"][state["dragging_point"]] = (x, y)
            # Show alignment guides
            img_copy = orig_image.copy()
            draw_alignment_guides(img_copy, state["points"], state["dragging_point"], (x, y))
            state["display_img"] = update_display_image(img_copy, state["points"], state["dragging_point"], (x, y))
            add_instructions(state["display_img"], state["instructions"])
            cv2.imshow(state["window_name"], state["display_img"])
            
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging_point"] = None

def update_display_image(base_image, points, active_idx=None, active_pos=None):
    """
    Updates the display image with all points and lines.
    
    Args:
        base_image: Base image to draw on
        points: List of all corner points
        active_idx: Index of actively dragged point (if any)
        active_pos: Current position of actively dragged point
        
    Returns:
        Updated image with points and lines
    """
    # Draw existing points and connections
    for i in range(len(points)):
        # Use active position for dragged point
        if i == active_idx and active_pos is not None:
            pt = active_pos
        else:
            pt = points[i]
            
        # Draw the point
        cv2.circle(base_image, pt, 5, (0, 255, 0), -1)
        
        # Draw label
        label = ["TL", "TR", "BR", "BL"][min(i, 3)]  # Top-left, Top-right, etc.
        cv2.putText(base_image, label, (pt[0] + 10, pt[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Connect points with lines if we have more than one point
        if len(points) > 1:
            # Get next point (looping back to start if needed)
            next_idx = (i + 1) % len(points)
            
            # Handle the case where the next point is being dragged
            if next_idx == active_idx and active_pos is not None:
                next_pt = active_pos
            else:
                next_pt = points[next_idx]
                
            # Draw the line
            cv2.line(base_image, pt, next_pt, (0, 255, 0), 2)
    
    return base_image

def add_instructions(image, instructions):
    """Add instructions to the image"""
    for i, text in enumerate(instructions):
        cv2.putText(image, text, (10, 30 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image

def draw_alignment_guides(image, points, active_idx, current_pos):
    """Draw guide lines to help with point alignment"""
    if len(points) < 2:
        return
    
    for i, point in enumerate(points):
        if i != active_idx:
            # Draw vertical alignment line
            if abs(current_pos[0] - point[0]) < 10:
                cv2.line(image, (point[0], 0), (point[0], image.shape[0]), 
                        (255, 255, 0), 1, cv2.LINE_AA)
            
            # Draw horizontal alignment line
            if abs(current_pos[1] - point[1]) < 10:
                cv2.line(image, (0, point[1]), (image.shape[1], point[1]), 
                        (255, 255, 0), 1, cv2.LINE_AA)

def get_user_points_and_crop(image_path, output_size=(800, 800)):
    """
    Allows the user to select 4 points on an image with dragging functionality
    and performs a perspective transform.

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
    orig_img = image.copy()
    
    # Set up the window with instructions
    window_name = "Select 4 corners by dragging"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, image.shape[1]), min(900, image.shape[0]))
    
    # Instructions overlay
    instructions = [
        "Click and drag to draw a rectangle around the chess board",
        "After drawing, drag corner points to adjust",
        "Press 'r' to reset",
        "Press 'c' to continue",
        "Press 'q' to quit"
    ]
    
    # Add instructions to the image
    display_img = add_instructions(display_img.copy(), instructions)
    
    # State dictionary to track interactive elements
    state = {
        "points": [],
        "display_img": display_img,
        "orig_img": orig_img,
        "drawing": False,
        "top_left": None,
        "window_name": window_name,
        "instructions": instructions,
        "dragging_point": None,  # Index of point being dragged
        "point_drag_radius": 10,  # Radius within which to detect point dragging
        "mode": "drawing",  # Current mode: "drawing" or "adjusting"
    }
    
    cv2.setMouseCallback(window_name, click_and_drag_points, state)
    
    while True:
        cv2.imshow(window_name, state["display_img"])
        key = cv2.waitKey(1) & 0xFF
        
        # Reset points if 'r' is pressed
        if key == ord('r'):
            state["points"] = []
            state["drawing"] = False
            state["top_left"] = None
            state["display_img"] = orig_img.copy()
            state["mode"] = "drawing"  # Reset mode to drawing
            # Re-add instructions
            add_instructions(state["display_img"], instructions)
            
        # Continue if 'c' is pressed and 4 points are selected
        elif key == ord('c') and len(state["points"]) == 4:
            break
            
        # Quit if 'q' is pressed
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None, None
    
    cv2.destroyAllWindows()
    
    if len(state["points"]) != 4:
        raise ValueError("Exactly 4 points must be selected.")
    
    # Order points in [top-left, top-right, bottom-right, bottom-left] order
    ordered_points = order_points(np.array(state["points"]))
    
    # Define the destination points (output rectangle)
    dst = np.array([
        [0, 0],                           # top-left
        [output_size[0], 0],              # top-right
        [output_size[0], output_size[1]], # bottom-right
        [0, output_size[1]]               # bottom-left
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
