import os
import cv2
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from utils.labels import convert_yolo_to_xy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
from typing import Optional, Sequence


COLOR_MAP = {
    0: (204, 153, 255),   
    1: (120, 240, 80),    # 
    2: (255, 100, 100),    #
    3: (255, 255, 0)      
}

def get_class_color(class_id: int, class_color_map: dict[int, tuple[int, int, int]]) -> tuple[int, int, int]:
    """
        Assign or retrieve a consistent RGB color for a given class ID.

        Args:
            class_id (int): The class ID for which to get or assign a color.
            class_color_map (dict[int, tuple[int, int, int]]): Dictionary mapping class IDs to RGB colors.

        Returns:
            tuple[int, int, int]: RGB color tuple associated with the class ID.
    """
    if class_color_map is None:
        class_color_map = COLOR_MAP

    if class_id not in class_color_map:
        class_color_map[class_id] = tuple(random.randint(50, 255) for _ in range(3))
    return class_color_map[class_id]


def visualize_bboxes(directory_path: str, class_color_map: dict[int, tuple[int, int, int]] | None = None, multiple_display: int = 1) -> None:
    """
        Visualizes bounding boxes on images with YOLO-style annotations.

        Args:
            directory_path (str): Path to directory containing 'images' and 'labels' folders or directly images/labels.
            class_color_map (dict[int, tuple[int, int, int]] | None): Optional mapping of class IDs to colors. Defaults to preset colors if None.
            multiple_display (int): Number of images to visualize. Defaults to 1.

        Returns:
            None: Displays images with bounding boxes using matplotlib.
    """

    class_color_map = {
        0: (204, 153, 255),   # màu tím nhạt
        1: (120, 240, 80),    # xanh lá
        2: (255, 100, 100),    # đỏ nhạt
        3: (255, 255, 0)      # vàng
    }

    try:
        image_path = os.path.join(directory_path, "images")
        txt_path = os.path.join(directory_path, "labels")
    except:
        image_path = directory_path
        txt_path = directory_path

    images = {}
    i = 0
    for filename in os.listdir(image_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_path, filename))
            img_height, img_width = img.shape[:2]
            filename = os.path.basename(filename)

            img_ = img.copy()
            try:
                with open(os.path.join(txt_path, filename.replace(".jpg", ".txt")), 'r') as f:
                    for line in f:
                        xmin, ymin, xmax, ymax, class_id = convert_yolo_to_xy(line.strip(), img_width, img_height)
                        color = get_class_color(class_id, class_color_map)
                        cv2.rectangle(img_, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(img_, str(class_id), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2, cv2.LINE_AA)
            except:
                print(f"Warning: No bounding box found for {filename}")
            images[filename] = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        i += 1
        if i == multiple_display:
            break

    if len(images) == 0:
        print("No images found in the directory.")
        return
    
    elif len(images) == 1:
        for name, img in images.items():
            plt.imshow(img)
            plt.axis('off')
            plt.title(name)
            plt.show()

    else:
        cols = min(3, len(images))
        rows = (len(images) + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 5 * rows))

        for i, (name, img) in enumerate(images.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(name)
        
        plt.tight_layout()
        plt.show()


def visualize_bboxes_distribution(
            directory_path: str | None = None,
            boxes: np.ndarray | None = None,
            anchors: np.ndarray | None = None,
            show_anchor: bool = False
        ) -> None:
    """
        Visualizes distribution of bounding box widths and heights and optionally anchor boxes.

        Args:
            directory_path (str | None): Path to directory containing YOLO label (.txt) files. Used if boxes and anchors are None.
            boxes (np.ndarray | None): Array of bounding boxes as (width, height). If None, loaded from directory_path.
            anchors (np.ndarray | None): Array of anchor boxes as (width, height). If None, computed from boxes.
            show_anchor (bool): Whether to display anchor boxes on the plot. Default is False.

        Returns:
            None: Displays a scatter plot using matplotlib.
    """

    # if boxes is None and anchors is None:
    #     boxes = load_yolo_wh(directory_path)
    #     anchors = kmeans_anchors(boxes, k=3)

    plt.scatter(boxes[:, 0], boxes[:, 1], c='lightblue', s=2, label='GT boxes')
    if show_anchor:
      plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=100, label='Anchors')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Anchor Boxes via K-means')
    plt.legend()
    plt.grid(True)
 
    plt.show()


def draw_cdf_from_histogram(image_path: str) -> None:
    """
        Draws the cumulative distribution function (CDF) from an image's grayscale histogram.

        Args:
            image_path (str): Path to the image file.

        Returns:
            None: Displays the CDF plot using matplotlib.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return

        # Calculate histogram
        hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])

        # Calculate CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize for plotting

        # Plot CDF
        plt.figure(figsize=(8, 6))
        plt.plot(cdf_normalized, color='b')
        plt.hist(img.flatten(), bins=256, color='gray', alpha=0.3)  # Overlay histogram

        plt.xlim([0, 256])
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Cumulative Frequency')
        plt.title('CDF of ' + image_path.split("/")[-1])
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

def draw_pixel_distribution(image_path: str) -> None:
    """
        Draws the pixel intensity distribution of an RGB image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            None: Displays histograms for the Red, Green, and Blue channels using matplotlib.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image from {image_path}")
            return

        # Flatten the image and separate color channels
        img_flat = img.reshape(-1, 3)
        r, g, b = img_flat[:, 0], img_flat[:, 1], img_flat[:, 2]


        # Create the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(r, bins=256, color='red', alpha=0.5, label='Red')
        plt.hist(g, bins=256, color='green', alpha=0.5, label='Green')
        plt.hist(b, bins=256, color='blue', alpha=0.5, label='Blue')

        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title('Pixel Distribution of ' + image_path.split("/")[-1])
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

def show_sample_from_loader(
        loader: DataLoader,
        class_names: list[str] | None = None,
        n_images: int = 4
    ) -> None:
    """
        Displays sample images from a PyTorch DataLoader with bounding boxes overlaid.

        Args:
            loader (DataLoader): PyTorch DataLoader providing batches of images and labels.
            class_names (list[str] | None): Optional list mapping class indices to human-readable labels.
            n_images (int): Number of images to display. Default is 4.

        Returns:
            None: Displays images with bounding boxes using matplotlib.
    """
    data_iter = iter(loader)
    for i in range(n_images):
        try:
            image, boxes, labels = next(data_iter)
        except StopIteration:
            print("Hết ảnh.")
            break

        img_np = image[0].permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]

        fig, ax = plt.subplots(1)
        ax.imshow(img_np)

        for bbox, label in zip(boxes[0], labels[0]):
            x_center, y_center, bw, bh = bbox.numpy()
            x = (x_center - bw / 2) * w
            y = (y_center - bh / 2) * h
            width = bw * w
            height = bh * h

            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            if class_names:
                ax.text(x, y - 5, class_names[int(label)], color='yellow', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.5))
            else:
                ax.text(x, y - 5, str(int(label.item())), color='yellow', fontsize=12,
                        bbox=dict(facecolor='black', alpha=0.5))

        plt.axis('off')
        plt.show()

def plot_class_distribution_after_copy_paste(
                                        dataset: object,
                                        class_names: Optional[Sequence[str]] = None,
                                        title: str = "Original Dataset",
                                        show_percentage: bool = True
                                    ) -> None:
    """
        Counts and visualizes the distribution of bounding boxes per class in a dataset after applying Copy-Paste augmentation.

        Args:
            dataset (object): Dataset object supporting indexing, returning tuples where the third element is a list or tensor of class labels.
            class_names (Optional[Sequence[str]]): Optional list of class names for labeling the plot. Defaults to None.
            title (str): Title of the bar plot. Defaults to "Original Dataset".
            show_percentage (bool): Whether to print class distribution percentages alongside counts. Defaults to True.

        Returns:
            None: Prints class distribution percentages and displays a bar plot of bounding box counts per class.
    """
    class_counter = Counter()

    for i in tqdm(range(len(dataset)), desc="Processing dataset with Copy-Paste"):
        try:
            _, _, labels = dataset[i]  # labels là list[int] hoặc Tensor
            class_counter.update(labels)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    total = sum(class_counter.values())
    classes = sorted(class_counter.keys())
    counts = [class_counter[c] for c in classes]
    class_labels = [class_names[c] if class_names and c < len(class_names) else f"Class {c}" for c in classes]

    if show_percentage:
        print("📊 Class distribution (percentage):")
        for cls_id, count in zip(classes, counts):
            label = class_labels[cls_id] if cls_id < len(class_labels) else f"Class {cls_id}"
            percent = 100 * count / total
            print(f"  {label:<10}: {count:>5} boxes ({percent:.2f}%)")

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Bounding Box Count")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    
def plot_transform_image(image_path, tech="gamma", **kwargs):

  if tech == "gamma":
    gamma = kwargs.get("gamma", 1.0)
    brightned_img = gamma_correction(image_path, gamma=gamma)
    title = "Gamma_Correction"

  elif tech == "clahe":
    brightned_img = apply_clahe(image_path)
    title = "CLAHE Enhanced"

  elif tech == "hsv":
    value = kwargs.get("value", 50)
    brightned_img = increase_brightness_hsv(image_path, value=value)
    title = "HSV_Adjusment"

  elif tech == "histogram":
    brightned_img = histogram_equalization_rgb(image_path)
    title = "Histogram_Equalization"

  elif tech == "linear":
    alpha = kwargs.get("alpha", 1.2)
    beta = kwargs.get("beta", 50)
    brightned_img = adjust_brightness(image_path, alpha=alpha, beta=beta)
    title = "Linear_Transformation"

  else:
    raise ValueError("Invalid technique. Choose from 'gamma', 'clahe', 'hsv', 'histogram', or 'linear'.")

  image = cv2.imread(image_path)
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.title("Original Image")
  plt.axis("off")

  plt.subplot(1, 2, 2)
  plt.imshow(brightned_img)
  plt.title(title)
  plt.axis("off")
  plt.show()

# Histogram Equalization
def histogram_equalization_rgb(image_path):
    """
    Applies histogram equalization to a color image using the YCrCb color space.

    This function enhances the contrast of the input image by performing histogram equalization 
    on the luminance (Y) channel, while preserving the chrominance (Cr, Cb) channels.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: The contrast-enhanced image in BGR format.
    """

    image = cv2.imread(image_path,)
    # Convert BGR to YCrCb
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Separate the color channels (Y, Cr, Cb)
    y, cr, cb = cv2.split(img_ycrcb)

    # Perform histogram equalization on the Y (luminance) channel
    y_eq = cv2.equalizeHist(y)

    # Merge the channels
    img_eq_ycrcb = cv2.merge((y_eq, cr, cb))

    # Convert the image back to the BGR color space (or RGB if required)
    img_equalized = cv2.cvtColor(img_eq_ycrcb, cv2.COLOR_YCrCb2BGR)

    #draw_histogram_cdf(img_equalized, title="Equalized Image")
    return img_equalized

# HSV Brightness Adjustment
def increase_brightness_hsv(image_path, value=50):
    """
    Increases the brightness of an image using the HSV color space.

    The function adjusts the V (value/brightness) channel of the image and clamps 
    pixel values to avoid overflow.

    Args:
        image_path (str): Path to the input image.
        value (int): Brightness increment value (default is 50).

    Returns:
        numpy.ndarray: The brightness-enhanced image in BGR format.
    """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.add(v, value)
    v[v > 255] = 255

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image_path):
    """
    Enhances the contrast of a color image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE is applied to the L (lightness) channel in the Lab color space to avoid over-amplifying noise.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: The contrast-enhanced image in BGR format.
    """

    image = cv2.imread(image_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

# Gamma Correction
def gamma_correction(image_path, gamma=1.0):
    """
    Adjusts image brightness using gamma correction.

    This method corrects image brightness non-linearly. Gamma < 1 brightens the image, 
    while gamma > 1 darkens it.

    Args:
        image_path (str): Path to the input image.
        gamma (float): Gamma value for correction (default is 1.0).

    Returns:
        numpy.ndarray: The gamma-corrected image.
    """

    image = cv2.imread(image_path)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Linear Brightness Adjustment
def adjust_brightness(image_path, alpha=1.2, beta=50):
    """
    Adjusts the brightness and contrast of an image linearly.

    This function modifies the pixel values using the formula:
        new_pixel = alpha * original_pixel + beta

    Args:
        image_path (str): Path to the input image.
        alpha (float): Contrast control (default is 1.2).
        beta (int): Brightness control (default is 50).

    Returns:
        numpy.ndarray: The adjusted image.
    """

    image = cv2.imread(image_path)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

