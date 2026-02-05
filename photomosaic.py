"""
Photomosaic Generator
Creates a square photomosaic with Windows 11 logo style.
Uses images from Bsod, Crash, Errors, and Out of Memory folders.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import math

# Configuration
TILE_SIZE = 400  # Large tiles for zoom detail
OUTPUT_SIZE = 3000  # Square output (will be adjusted based on tile count)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
BLACK_THRESHOLD = 30  # Pixels darker than this are considered black (no tile)

# Windows 11 logo colors (with gradient effect)
WINDOWS_COLORS = {
    'red': (243, 83, 37),      # Top-left quadrant
    'green': (129, 188, 6),    # Top-right quadrant
    'blue': (5, 166, 240),     # Bottom-left quadrant
    'yellow': (255, 186, 8),   # Bottom-right quadrant
    'black': (0, 0, 0),        # Background
}

def create_windows_logo(size):
    """Generate a Windows 11 style logo: black bg with 4 gradient-shaded squares."""
    img = Image.new('RGB', (size, size), WINDOWS_COLORS['black'])
    
    # Square layout: 4 quadrants with gaps
    margin = int(size * 0.12)  # Border around the logo
    gap = int(size * 0.03)     # Gap between quadrants
    
    total_inner = size - 2 * margin - gap
    quad_size = total_inner // 2
    
    quadrants = [
        ('red', margin, margin),                                    # Top-left
        ('green', margin + quad_size + gap, margin),                # Top-right
        ('blue', margin, margin + quad_size + gap),                 # Bottom-left
        ('yellow', margin + quad_size + gap, margin + quad_size + gap),  # Bottom-right
    ]
    
    for color_name, x, y in quadrants:
        base_color = WINDOWS_COLORS[color_name]
        # Create gradient within each quadrant (lighter top-left to darker bottom-right)
        for py in range(quad_size):
            for px in range(quad_size):
                # Calculate gradient factor (0.0 at top-left, 1.0 at bottom-right)
                gradient = (px + py) / (2 * quad_size)
                # Darken towards bottom-right
                factor = 1.0 - (gradient * 0.3)  # 30% darkening max
                r = int(base_color[0] * factor)
                g = int(base_color[1] * factor)
                b = int(base_color[2] * factor)
                img.putpixel((x + px, y + py), (r, g, b))
    
    return img

def get_average_color(image):
    """Calculate the average RGB color of an image."""
    img = image.convert('RGB')
    pixels = img.tobytes()
    total_pixels = len(pixels) // 3
    if total_pixels == 0:
        return (0, 0, 0)
    
    r = sum(pixels[i] for i in range(0, len(pixels), 3)) // total_pixels
    g = sum(pixels[i] for i in range(1, len(pixels), 3)) // total_pixels
    b = sum(pixels[i] for i in range(2, len(pixels), 3)) // total_pixels
    return (r, g, b)

def color_distance(c1, c2):
    """Calculate Euclidean distance between two RGB colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def load_tile_images(source_dirs, tile_size):
    """Load all images from source directories, resize to tiles, and calculate average colors."""
    tiles = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"  Skipping (not found): {source_path}")
            continue
            
        print(f"Loading images from: {source_path}")
        count = 0
        
        for file_path in source_path.iterdir():
            # Skip directories and non-image files
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            try:
                img = Image.open(file_path)
                img = img.convert('RGB')
                # Resize to tile size (crop to square first for better appearance)
                min_dim = min(img.size)
                left = (img.width - min_dim) // 2
                top = (img.height - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                
                avg_color = get_average_color(img)
                tiles.append({
                    'image': img,
                    'color': avg_color,
                    'path': file_path.name
                })
                count += 1
            except Exception as e:
                print(f"  Skipping {file_path.name}: {e}")
        
        print(f"  Loaded {count} images from {source_path.name}")
    
    print(f"Total: {len(tiles)} tile images")
    return tiles

def find_best_tile(target_color, tiles):
    """Find the tile with the closest average color to the target."""
    best_tile = None
    best_distance = float('inf')
    
    for tile in tiles:
        dist = color_distance(target_color, tile['color'])
        if dist < best_distance:
            best_distance = dist
            best_tile = tile
    
    return best_tile

def assign_tiles_ensure_all_used(target_colors, tiles):
    """
    Assign tiles to positions ensuring every tile is used at least once,
    and no tile is used more than MAX_REUSE times for diversity.
    Spreads similar colors across each quadrant to avoid clustering.
    """
    import random
    random.seed(42)  # Reproducible results
    
    # Filter out black positions (those stay empty)
    positions = [(r, c, color) for r, c, color in target_colors 
                 if max(color) > BLACK_THRESHOLD]
    num_positions = len(positions)
    num_tiles = len(tiles)
    
    if num_positions == 0:
        print("  Warning: No non-black positions found!")
        return {}
    
    # Calculate max reuse to ensure diversity
    max_reuse = max(2, (num_positions + num_tiles - 1) // num_tiles)
    
    print(f"  Assigning {num_tiles} tiles to {num_positions} positions")
    print(f"  Max reuse per tile: {max_reuse}")
    
    # Group positions by quadrant based on their location in the image
    # The logo is divided into 4 quadrants with a gap in the middle
    def get_quadrant_by_position(row, col, grid_size):
        margin = int(grid_size * 0.12)
        gap_center = grid_size // 2
        
        # Determine which quadrant based on row/col position
        if row < gap_center and col < gap_center:
            return 'red'      # Top-left
        elif row < gap_center and col >= gap_center:
            return 'green'    # Top-right
        elif row >= gap_center and col < gap_center:
            return 'blue'     # Bottom-left
        else:
            return 'yellow'   # Bottom-right
    
    # Get grid size from positions
    max_row = max(p[0] for p in positions)
    max_col = max(p[1] for p in positions)
    grid_size = max(max_row, max_col) + 1
    
    quadrant_positions = {'red': [], 'green': [], 'blue': [], 'yellow': []}
    for pos in positions:
        r, c, color = pos
        q = get_quadrant_by_position(r, c, grid_size)
        quadrant_positions[q].append((r, c, color))
    
    # For each quadrant, shuffle positions for random placement
    for q in quadrant_positions:
        random.shuffle(quadrant_positions[q])
    
    print(f"  Positions per quadrant: red={len(quadrant_positions['red'])}, "
          f"green={len(quadrant_positions['green'])}, blue={len(quadrant_positions['blue'])}, "
          f"yellow={len(quadrant_positions['yellow'])}")
    
    # Calculate how well each tile matches each quadrant
    def quadrant_match_score(tile_color, quadrant):
        base_colors = {
            'red': WINDOWS_COLORS['red'],
            'green': WINDOWS_COLORS['green'],
            'blue': WINDOWS_COLORS['blue'],
            'yellow': WINDOWS_COLORS['yellow'],
        }
        return color_distance(tile_color, base_colors[quadrant])
    
    # Score each tile for each quadrant
    tile_scores = []
    for tile_idx, tile in enumerate(tiles):
        scores = {q: quadrant_match_score(tile['color'], q) for q in quadrant_positions}
        best_quadrant = min(scores, key=scores.get)
        tile_scores.append((scores[best_quadrant], tile_idx, best_quadrant, scores))
    
    # Sort tiles by their best match score
    tile_scores.sort()
    
    # Assign tiles to quadrants
    tile_usage = {i: 0 for i in range(num_tiles)}
    quadrant_tiles = {'red': [], 'green': [], 'blue': [], 'yellow': []}
    tiles_assigned = set()
    
    # First pass: assign each tile to its best quadrant (if space available)
    for _, tile_idx, best_q, scores in tile_scores:
        if tile_idx in tiles_assigned:
            continue
        
        # Try best quadrant first, then others in order of match quality
        sorted_quads = sorted(scores.keys(), key=lambda q: scores[q])
        for q in sorted_quads:
            if len(quadrant_tiles[q]) < len(quadrant_positions[q]):
                quadrant_tiles[q].append(tile_idx)
                tiles_assigned.add(tile_idx)
                tile_usage[tile_idx] += 1
                break
    
    print(f"  First pass: {len(tiles_assigned)} tiles distributed to quadrants")
    
    # Second pass: fill remaining slots with reuse
    for q in quadrant_positions:
        slots_remaining = len(quadrant_positions[q]) - len(quadrant_tiles[q])
        if slots_remaining > 0:
            # Get tiles sorted by match to this quadrant
            available = [(quadrant_match_score(tiles[i]['color'], q), i) 
                        for i in range(num_tiles) if tile_usage[i] < max_reuse]
            available.sort()
            
            # Shuffle tiles with similar scores to spread them out
            for score, tile_idx in available:
                if slots_remaining <= 0:
                    break
                if tile_usage[tile_idx] < max_reuse:
                    quadrant_tiles[q].append(tile_idx)
                    tile_usage[tile_idx] += 1
                    slots_remaining -= 1
    
    # Shuffle tiles within each quadrant for even distribution
    for q in quadrant_tiles:
        random.shuffle(quadrant_tiles[q])
    
    # Create final assignments by pairing shuffled tiles with shuffled positions
    assignments = {}
    for q in quadrant_positions:
        for i, (r, c, _) in enumerate(quadrant_positions[q]):
            if i < len(quadrant_tiles[q]):
                tile_idx = quadrant_tiles[q][i]
                assignments[(r, c)] = tiles[tile_idx]
    
    # Print usage stats
    usage_counts = [v for v in tile_usage.values() if v > 0]
    if usage_counts:
        print(f"  Usage stats: min={min(usage_counts)}, max={max(usage_counts)}, avg={sum(usage_counts)/len(usage_counts):.1f}")
    
    return assignments

def create_photomosaic(target_image, tiles, output_path, output_size, tile_size):
    """Create a photomosaic from the target image using the tile images."""
    if isinstance(target_image, (str, Path)):
        print(f"Loading target image: {target_image}")
        target = Image.open(target_image).convert('RGB')
    else:
        print("Using generated target image")
        target = target_image
    
    # Calculate grid dimensions (square)
    grid_size = output_size // tile_size
    
    # Resize target to match grid (each pixel will map to one tile)
    target_small = target.resize((grid_size, grid_size), Image.Resampling.LANCZOS)
    
    print(f"Creating mosaic: {grid_size}x{grid_size} tiles ({grid_size * tile_size}x{grid_size * tile_size} pixels)")
    
    # Build list of target colors for all positions
    target_colors = []
    for row in range(grid_size):
        for col in range(grid_size):
            color = target_small.getpixel((col, row))
            target_colors.append((row, col, color))
    
    # Assign tiles ensuring every tile is used at least once
    assignments = assign_tiles_ensure_all_used(target_colors, tiles)
    
    # Create output image (black background)
    final_size = grid_size * tile_size
    mosaic = Image.new('RGB', (final_size, final_size), (0, 0, 0))
    
    print("  Compositing mosaic...")
    for (row, col), tile in assignments.items():
        x = col * tile_size
        y = row * tile_size
        mosaic.paste(tile['image'], (x, y))
    
    # Save the mosaic
    mosaic.save(output_path, quality=95)
    print(f"Saved mosaic to: {output_path}")
    
    # Also save the target for reference
    target_ref_path = output_path.parent / "target_reference.png"
    target.save(target_ref_path)
    print(f"Saved target reference to: {target_ref_path}")
    
    return mosaic

def main():
    # Paths
    script_dir = Path(__file__).parent
    bsod_dir = script_dir.parent  # c:\Talks\funnies\Bsod
    funnies_dir = bsod_dir.parent  # c:\Talks\funnies
    output_path = script_dir / "photomosaic_output.jpg"
    
    # Source directories for images
    source_dirs = [
        bsod_dir,                            # Main Bsod folder
        funnies_dir / "Crash",               # Crash folder
        funnies_dir / "Errors",              # Errors folder
        funnies_dir / "Out of Memory",       # Out of Memory folder
    ]
    
    # Load tile images from all source directories
    print("Loading tile images...")
    tiles = load_tile_images(source_dirs, TILE_SIZE)
    
    if not tiles:
        print("Error: No tile images found!")
        return
    
    # Calculate grid size based on number of tiles
    # The logo has ~65% colored area (4 squares), rest is black margin/cross
    # We want each tile used ~1-2 times, so colored positions ≈ 1.5x tiles
    num_tiles = len(tiles)
    target_colored_positions = int(num_tiles * 1.5)
    # Total grid positions = colored / 0.65
    total_positions = int(target_colored_positions / 0.65)
    grid_size = int(math.sqrt(total_positions))
    output_size = grid_size * TILE_SIZE
    
    print()
    print(f"Tiles available: {num_tiles}")
    print(f"Grid size: {grid_size}x{grid_size} = {grid_size**2} total positions")
    print(f"Output size: {output_size}x{output_size} pixels (square)")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print()
    
    # Generate Windows 11 style logo as target
    print("Generating Windows 11 style logo as target image")
    target_image = create_windows_logo(output_size)
    
    # Create the photomosaic
    create_photomosaic(target_image, tiles, output_path, output_size, TILE_SIZE)
    
    print()
    print("Done! Photomosaic created successfully.")
    print(f"All {len(tiles)} images were used at least once.")

if __name__ == "__main__":
    main()
