from roboflow import Roboflow
import cv2
import os
import json
from typing import Dict, List, Optional, Tuple, Union

def detect_chess_pieces(
    image_path: str, 
    output_dir: str, 
    api_key: str, 
    model_id: str, 
    version: int,
    confidence_threshold: float = 0.4,
    overlap_threshold: float = 0.3,
    draw_detections: bool = True,
    save_json: bool = True
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Detect chess pieces on the given image using the Roboflow model.

    Args:
        image_path (str): Path to the input image (cropped chessboard).
        output_dir (str): Directory to save the output images and coordinates.
        api_key (str): Roboflow API key.
        model_id (str): Roboflow model ID.
        version (int): Version of the Roboflow model.
        confidence_threshold (float, optional): Confidence threshold for detections (0-1). Defaults to 0.4.
        overlap_threshold (float, optional): Overlap threshold for non-max suppression (0-1). Defaults to 0.3.
        draw_detections (bool, optional): Whether to draw and save detection visualization. Defaults to True.
        save_json (bool, optional): Whether to save results as JSON. Defaults to True.

    Returns:
        Dict[str, Union[str, List[Dict]]]: Dictionary containing detection results and output file paths.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate the input image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Initialize Roboflow model with error handling
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(model_id)
        model = project.version(version).model
    except Exception as e:
        raise ConnectionError(f"Failed to initialize Roboflow model: {str(e)}")

    # Perform inference on the image
    try:
        # Convert confidence threshold from 0-1 to 0-100 range for Roboflow API
        confidence_percent = int(confidence_threshold * 100)
        overlap_percent = int(overlap_threshold * 100)
        predictions = model.predict(image_path, confidence=confidence_percent, overlap=overlap_percent).json()
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

    # Load the image for drawing detections
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")

    # Process predictions
    detected_pieces = []
    colors = {
        "white": (0, 255, 0),  # Green for white pieces
        "black": (0, 0, 255)   # Red for black pieces
    }
    
    # Define base filename for outputs using the input filename
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    for prediction in predictions['predictions']:
        x, y = prediction['x'], prediction['y']
        width, height = prediction['width'], prediction['height']
        label = prediction['class']
        confidence = prediction['confidence']
        
        # Determine piece color from label (assuming format like "white_pawn", "black_queen")
        color_name = label.split('_')[0] if '_' in label else "unknown"
        piece_type = label.split('_')[1] if '_' in label else label
        
        # Calculate bounding box coordinates
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)
        
        # Store detection information
        piece_info = {
            "label": label,
            "piece_type": piece_type,
            "color": color_name,
            "confidence": confidence,
            "center": (x, y),
            "box": [x1, y1, x2, y2],
            "width": width,
            "height": height
        }
        detected_pieces.append(piece_info)
        
        # Draw bounding box if requested
        if draw_detections:
            box_color = colors.get(color_name, (255, 255, 0))  # Default to yellow if color unknown
            text_color = (255, 255, 255)  # White text
            
            # Draw filled rectangle with transparency for better text visibility
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            
            # Draw border
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            
            # Add text with better formatting
            label_text = f"{piece_type} ({confidence:.2f})"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Create a filled background for text
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 5, y1), box_color, -1)
            cv2.putText(image, label_text, (x1 + 2, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Create result dictionary
    result = {
        "detected_pieces": detected_pieces,
        "piece_count": len(detected_pieces),
        "piece_distribution": _count_pieces_by_type(detected_pieces),
        "output_files": {}
    }
    
    # Save the visualization image if requested
    if draw_detections:
        detected_image_path = os.path.join(output_dir, f"{base_filename}_detected.jpg")
        cv2.imwrite(detected_image_path, image)
        result["output_files"]["visualization"] = detected_image_path
    
    # Save results in multiple formats
    if save_json:
        json_path = os.path.join(output_dir, f"{base_filename}_detections.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        result["output_files"]["json"] = json_path
    
    # Always save text version for backward compatibility
    txt_path = os.path.join(output_dir, f"{base_filename}_coordinates.txt")
    with open(txt_path, 'w') as f:
        for piece in detected_pieces:
            f.write(f"{piece['label']} ({piece['confidence']:.2f}): {piece['box']}\n")
    result["output_files"]["text"] = txt_path
    
    print(f"Detected {len(detected_pieces)} chess pieces")
    for file_type, file_path in result["output_files"].items():
        print(f"Saved {file_type} output to: {file_path}")
    
    return result

def _count_pieces_by_type(detected_pieces: List[Dict]) -> Dict[str, int]:
    """
    Helper function to count pieces by type and color.
    
    Args:
        detected_pieces (List[Dict]): List of detected piece information.
        
    Returns:
        Dict[str, int]: Dictionary with counts of each piece type.
    """
    counts = {}
    for piece in detected_pieces:
        label = piece["label"]
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    
    return counts