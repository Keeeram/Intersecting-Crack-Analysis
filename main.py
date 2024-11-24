import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label
from ultralytics import YOLO
from PIL import Image
import matplotlib
#显示中文
matplotlib.rc("font", family='Microsoft YaHei')

#读取YOLO模型
def trained_model(model_path):
    model = YOLO(model_path)
    return model
#调用模型
def perform_yolo_inference(model, image_path):
    results = model(image_path)
    return results
def extract_masks(results):
    for result in results:
        masks = result.masks
        return masks.xy
def create_binary_image(masks_xy, image_shape):
    binary_image = np.zeros(image_shape, dtype=np.uint8)
    for mask in masks_xy:
        polygon = [(int(point[0]), int(point[1])) for point in mask]
        cv2.fillPoly(binary_image, [np.array(polygon)], 255)
    return binary_image

def read_and_binarize_image(image_path):
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    _, binary_im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    return binary_im, im

def extract_skeleton(binary_im):
    skeleton = skeletonize(binary_im // 255, method='lee')
    return skeleton

def find_points(skeleton, min_distance=5):
    points = np.zeros_like(skeleton)
    confirmed_points = []

    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1:
                num_neighbors = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2])
                if num_neighbors == 4:
                    confirmed_points.append((y, x))

    ignore_points = []
    for fork_candidate in confirmed_points:
        if fork_candidate in ignore_points:
            continue
        nearby_points = [pt for pt in confirmed_points if np.linalg.norm(np.array(pt) - np.array(fork_candidate)) < min_distance]
        if nearby_points:
            centroid = np.mean(nearby_points, axis=0).astype(int)
            points[centroid[0], centroid[1]] = 1
            ignore_points.extend(nearby_points)
    return points

def find_end_points(skeleton):
    end_points = np.zeros_like(skeleton)
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1:
                num_neighbors = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2])
                if num_neighbors == 2:
                    end_points[y, x] = 1
    return end_points

def calculate_branch_length_from_fork(skeleton, fork_point):
    y, x = fork_point
    length = 0
    visited = set()
    stack = [(y, x)]

    directions = [
        (-1, 0), (1, 0),  #垂直
        (0, -1), (0, 1),  #水平
        (-1, -1), (-1, 1),  #对角线
        (1, -1), (1, 1)
    ]

    while stack:
        current = stack.pop()
        cy, cx = current
        visited.add((cy, cx))

        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                if skeleton[ny, nx] == 1 and (ny, nx) not in visited:
                    length += 1 if abs(dy) + abs(dx) == 1 else np.sqrt(2)
                    stack.append((ny, nx))
    return length
def calculate_average_branch_length(skeleton, fork_points):
    total_length = 0
    num_forks = 0
    fork_coords = np.argwhere(fork_points)

    for fork in fork_coords:
        length = calculate_branch_length_from_fork(skeleton, fork)
        total_length += length
        num_forks += 1

    average_length = total_length / num_forks if num_forks > 0 else 0
    return average_length

def extend_branches(skeleton, fork_points):
    labeled_branches = label(skeleton)
    num_branches = np.max(labeled_branches)
    branch_images = []
    for branch_id in range(1, num_branches + 1):
        branch_image = np.zeros_like(skeleton)
        branch_pixels = np.where(labeled_branches == branch_id)
        branch_image[branch_pixels] = 1
        branch_images.append(branch_image)

    return branch_images

def color_branches(branch_images):
    color_skeleton = np.zeros((branch_images[0].shape[0], branch_images[0].shape[1], 3), dtype=np.uint8)
    for idx, branch_image in enumerate(branch_images):
        color_skeleton[branch_image > 0] = [255, 255, 0]  # Yellow branches
    return color_skeleton

def display_binary_and_skeleton_with_points(binary_im, skeleton, fork_coords):
    plt.figure(figsize=(6, 5))
    plt.imshow(binary_im, cmap='cividis')
    for y, x in fork_coords:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    plt.title('Binary Image with points')
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(skeleton, cmap='cividis')
    for y, x in fork_coords:
        plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    plt.title('Skeleton Image with points')
    plt.show()

def annotate_fork_points_on_new_image(rgb_image_path, fork_coords):
    # 读取新的RGB图像
    rgb_image = cv2.imread(rgb_image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"RGB image at {rgb_image_path} not found.")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb_image)
    for y, x in fork_coords:
        plt.plot(x, y, 'yo', markersize=5)  # 黄色圆圈表示交叉点

    plt.title('intersection points')
    plt.show()

def main(yolo_model_path, rgb_image_path):
    # Load YOLO model and perform inference
    model = trained_model(yolo_model_path)
    results = perform_yolo_inference(model, rgb_image_path)

    # Extract masks and create a binary image
    masks_xy = extract_masks(results)
    image_shape = Image.open(rgb_image_path).size
    binary_image = create_binary_image(masks_xy, image_shape[::-1])  # Reverse size for OpenCV

    skeleton = extract_skeleton(binary_image)
    points = find_points(skeleton)
    end_points = find_end_points(skeleton)

    average_length = calculate_average_branch_length(skeleton, points)
    print("分支长度:", average_length)

    branch_images = extend_branches(skeleton, points)

    display_binary_and_skeleton_with_points(binary_image, skeleton, np.argwhere(points))

if __name__ == "__main__":
    yolo_model_path = r''
    rgb_image_path = r''
    main(yolo_model_path, rgb_image_path)