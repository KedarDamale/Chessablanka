import cv2
import numpy as np

def get_square_coordinates(board_size):
    """
    Generate coordinates for all squares on the chessboard
    
    Args:
        board_size (tuple): A tuple containing (width, height) of the board image
        
    Returns:
        dict: Dictionary mapping square names (e.g., 'A1') to their coordinates
    """
    square_coords = {}
    grid_size = board_size[0] // 8  # Assuming square board
    letters = "ABCDEFGH"
    
    for i in range(8):
        for j in range(8):
            square_name = f"{letters[j]}{8 - i}"
            x1, y1 = j * grid_size, i * grid_size
            x2, y2 = (j + 1) * grid_size, (i + 1) * grid_size
            # Store coordinates as four corners (top-left, top-right, bottom-right, bottom-left)
            square_coords[square_name] = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    return square_coords

def draw_chess_grid(frame, board_size, square_coords=None, line_thickness=2, text_scale=0.6, 
                    line_color=(255, 255, 255), text_color=(255, 255, 255), fill_squares=True):
    """
    Draw a chess grid on the frame and label each square
    
    Args:
        frame (numpy.ndarray): The image frame to draw on
        board_size (tuple): A tuple containing (width, height) of the board
        square_coords (dict, optional): Dictionary of square coordinates, if None will be generated
        line_thickness (int, optional): Thickness of grid lines
        text_scale (float, optional): Scale factor for text size
        line_color (tuple, optional): BGR color for grid lines
        text_color (tuple, optional): BGR color for square labels
        fill_squares (bool, optional): If True, fills squares with alternating colors
        
    Returns:
        numpy.ndarray: Frame with chess grid drawn on it
    """
    if square_coords is None:
        square_coords = get_square_coordinates(board_size)
    
    grid_size = board_size[0] // 8
    letters = "ABCDEFGH"
    
    # Create a copy of the frame to avoid modifying the original
    result = frame.copy()
    
    # Fill squares with alternating colors if requested
    if fill_squares:
        for i in range(8):
            for j in range(8):
                square_name = f"{letters[j]}{8 - i}"
                coords = square_coords[square_name]
                # Check if square should be dark or light
                is_dark = (i + j) % 2 == 1
                fill_color = (120, 120, 120, 128) if is_dark else None  # Semi-transparent gray for dark squares
                
                if is_dark:
                    # Create polygon points and fill
                    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                    overlay = result.copy()
                    cv2.fillPoly(overlay, [pts], fill_color[:3])
                    # Apply transparency
                    alpha = fill_color[3] / 255.0
                    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    # Draw grid lines
    for i in range(9):
        # Horizontal lines
        cv2.line(result, (0, i * grid_size), (board_size[0], i * grid_size), line_color, line_thickness)
        # Vertical lines
        cv2.line(result, (i * grid_size, 0), (i * grid_size, board_size[1]), line_color, line_thickness)
    
    # Label squares
    for i in range(8):
        for j in range(8):
            square_name = f"{letters[j]}{8 - i}"
            x1, y1 = j * grid_size, i * grid_size
            cv2.putText(result, square_name, (x1 + 10, y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, line_thickness)
    
    return result

def highlight_square(frame, square_name, board_size, color=(0, 255, 0), thickness=3):
    """
    Highlight a specific square on the chess grid
    
    Args:
        frame (numpy.ndarray): The image frame to draw on
        square_name (str): Chess notation of the square (e.g., 'E4')
        board_size (tuple): A tuple containing (width, height) of the board
        color (tuple, optional): BGR color for the highlight
        thickness (int, optional): Thickness of the highlight border
        
    Returns:
        numpy.ndarray: Frame with highlighted square
    """
    square_coords = get_square_coordinates(board_size)
    
    if square_name in square_coords:
        pts = np.array(square_coords[square_name], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, color, thickness)
    
    return frame

def save_square_coordinates(square_coords, output_path):
    """
    Save the square coordinates to a file.

    Args:
        square_coords (dict): Dictionary of square names and their coordinates.
        output_path (str): Path to save the coordinates file.
    """
    with open(output_path, 'w') as f:
        for square_name, coords in square_coords.items():
            coords_str = ', '.join([f"({x}, {y})" for x, y in coords])
            f.write(f"{square_name}: {coords_str}\n")
    print(f"Square coordinates saved to '{output_path}'")