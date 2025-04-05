import json

def map_pieces_to_squares(piece_coordinates_path, square_coordinates_path):
    """
    Map chess pieces to their respective squares based on coordinates.

    Args:
        piece_coordinates_path (str): Path to the file containing piece coordinates.
        square_coordinates_path (str): Path to the file containing square coordinates.

    Returns:
        dict: A dictionary mapping squares to the pieces on them.
    """
    # Load piece coordinates
    with open(piece_coordinates_path, 'r') as f:
        piece_data = []
        for line in f:
            label, rest = line.split(" (")
            confidence, box = rest.split("): ")
            confidence = float(confidence)
            box = eval(box.strip())
            piece_data.append({
                "label": label,
                "confidence": confidence,
                "box": box,
                "center": ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            })

    # Load square coordinates
    square_coords = {}
    with open(square_coordinates_path, 'r') as f:
        for line in f:
            square, coords = line.split(": ")
            coords = eval(coords.strip())
            square_coords[square] = coords

    # Map pieces to squares
    piece_to_square_mapping = {}
    unmapped_pieces = []  # Track pieces that are not mapped to any square
    for piece in piece_data:
        piece_center = piece["center"]
        mapped = False
        for square, coords in square_coords.items():
            x1, y1 = coords[0]
            x2, y2 = coords[2]
            if x1 <= piece_center[0] <= x2 and y1 <= piece_center[1] <= y2:
                piece_to_square_mapping[square] = piece["label"]
                mapped = True
                break
        if not mapped:
            unmapped_pieces.append(piece)

    # Log unmapped pieces
    if unmapped_pieces:
        print("\nUnmapped Pieces:")
        for piece in unmapped_pieces:
            print(f"Piece '{piece['label']}' at center {piece['center']} could not be mapped to any square.")

    return piece_to_square_mapping