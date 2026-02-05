# BSOD Photomosaic

A Windows 11 logo photomosaic created from 533 images of Blue Screens of Death (BSODs), crashes, errors, and out-of-memory screens captured in the wild.

![Photomosaic Output](photomosaic_output.jpg)

## About

This mosaic recreates the Windows 11 logo using photos of Windows crashes and errors found on public displays around the world - airports, ATMs, billboards, kiosks, restaurants, and more.

- **533 unique images** used as tiles
- **14,000 × 14,000 pixels** (square format)
- **400 × 400 pixel tiles** for zoom detail
- Each image used 1-2 times for variety
- Pure black margins and cross separating the four colored quadrants

## How It Works

The Python script:

1. Loads images from multiple source folders (BSOD, Crash, Errors, Out of Memory)
2. Resizes each to a square tile and calculates its average color
3. Generates a Windows 11 style target logo with gradient-shaded quadrants
4. Assigns tiles to quadrants based on color matching
5. Shuffles tile placement randomly within each quadrant for even distribution
6. Composites the final mosaic with black background

## Usage

```bash
pip install Pillow
python photomosaic.py
```

The script expects source images in the parent directory and sibling folders (`../Crash`, `../Errors`, `../Out of Memory`).

## Output

- `photomosaic_output.jpg` - The final mosaic image
- `target_reference.png` - The target Windows logo used as the template

## Requirements

- Python 3.x
- Pillow (PIL)

## Author

Created by Mark Russinovich
