import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Supported formats list
SUPPORTED_FORMATS = [
    ".bmp", ".dib",  # Windows bitmaps
    ".jpeg", ".jpg", ".jpe",  # JPEG files
    ".jp2",  # JPEG 2000 files
    ".png",  # Portable Network Graphics
    ".webp",  # WebP
    ".tiff", ".tif",  # TIFF files
    ".hdr", ".pic"  # Radiance HDR
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-5s] %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S"
)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from the given path using OpenCV.

    :param path: Path to the image file.
    :return: The loaded image as a numpy array.
    """
    with open(path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def save_image(path: str, image: np.ndarray, ext: str = ".jpg") -> None:
    """
    Save an image to the given path in the specified format.

    :param path: Path to save the image.
    :param image: Image as a numpy array.
    :param ext: Image file extension/format (default is ".jpg").
    """
    success, encoded_image = cv2.imencode(ext, image)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded_image)


def create_blurred_edges(
        image: np.ndarray,
        target_size: Tuple[int, int],
        padding_percent: float = 0.05
) -> np.ndarray:
    """
    Create a blurred background from the edges of the original image.

    :param image: Original image.
    :param target_size: Target size for the blurred background.
    :param padding_percent: Percentage of padding to be used for creating the blurred edges.
    :return: Blurred background as a numpy array.
    """
    h, w, _ = image.shape
    aspect_ratio = w / h

    if aspect_ratio > (target_size[0] / target_size[1]):
        top_crop = int(h * padding_percent)
        bottom_crop = h - top_crop
        top_part = image[:top_crop, :]
        bottom_part = image[bottom_crop:, :]
        edges_image = np.vstack((top_part, bottom_part))
    else:
        left_crop = int(w * padding_percent)
        right_crop = w - left_crop
        left_part = image[:, :left_crop]
        right_part = image[:, right_crop:]
        edges_image = np.hstack((left_part, right_part))

    blurred_background = cv2.resize(edges_image, target_size)
    blurred_background = cv2.GaussianBlur(blurred_background, (201, 201), 0)

    return blurred_background


def resize_with_smart_blur_background(
        image: np.ndarray,
        target_size: Tuple[int, int] = (1920, 1080),
        padding_percent: float = 0.05
) -> np.ndarray:
    """
    Resize the image and place it on a blurred background.

    :param image: Original image.
    :param target_size: Target size for the output image.
    :param padding_percent: Percentage of padding for the blurred edges.
    :return: Image with blurred background as a numpy array.
    """
    h, w, _ = image.shape
    aspect_ratio = w / h

    if aspect_ratio > (target_size[0] / target_size[1]):
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = target_size[1]
        new_w = int(new_h * aspect_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))
    blurred_background = create_blurred_edges(image, target_size, padding_percent)

    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    blurred_background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return blurred_background


def process_folder(
        input_folder: Path,
        output_folder: Path,
        ext: Optional[str] = None,
        resolution: Tuple[int, int] = (1920, 1080)
) -> None:
    """
    Process all supported images in the input folder, resize them, and apply blurred background.

    :param input_folder: Folder containing the images to process.
    :param output_folder: Folder to save the processed images.
    :param ext: Optional output format (file extension).
    :param resolution: Resolution for the output images.
    """
    files_to_process = [file for file in input_folder.iterdir() if file.suffix.lower() in SUPPORTED_FORMATS]
    logging.info(f"Found {len(files_to_process)} files to process.")

    for filename in files_to_process:
        new_ext = ext or filename.suffix
        try:
            image = load_image(str(filename))
            result = resize_with_smart_blur_background(image, target_size=resolution)
            output_path = output_folder / filename.with_suffix(new_ext).name
            if output_path.exists() or filename.suffix.lower() != new_ext.lower():
                output_path = output_folder / (filename.stem + new_ext)
            save_image(str(output_path), result, ext=new_ext)
            logging.info(f"Successfully processed: {output_path}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e).strip()}")


def main() -> None:
    """
    Main entry point for the script, handles argument parsing and initial checks.
    """
    parser = argparse.ArgumentParser(description="Batch process images by resizing and applying blurred background.")
    parser.add_argument("src", type=str, help="Source folder containing images")
    parser.add_argument("dst", type=str, help="Destination folder for processed images")
    parser.add_argument("--format", type=str, default=None, help="Optional output format (e.g., .jpg, .png)")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080),
                        help="Output resolution (width height)")

    args = parser.parse_args()

    input_folder = Path(args.src)
    output_folder = Path(args.dst)

    # Check if the source folder exists
    if not input_folder.exists() or not input_folder.is_dir():
        logging.error(f"Source folder '{input_folder}' does not exist or is not a directory.")
        return

    # Check if format is in SUPPORTED_FORMATS
    if args.format and args.format.lower() not in SUPPORTED_FORMATS:
        logging.error(f"Unsupported format '{args.format}'. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")
        return

    # Create the destination folder if it doesn't exist
    if output_folder.exists():
        if not output_folder.is_dir():
            logging.error(f"Destination folder '{output_folder}' is not a directory.")
            return
    else:
        output_folder.mkdir(parents=True)

    logging.info(f"Starting processing with source: {input_folder}, destination: {output_folder}, "
                 f"format: {args.format}, resolution: {args.resolution}")

    process_folder(input_folder, output_folder, ext=args.format, resolution=args.resolution)


if __name__ == "__main__":
    main()
