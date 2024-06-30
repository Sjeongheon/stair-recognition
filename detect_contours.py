import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image, low_threshold=40, high_threshold=120):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Canny(blur, low_threshold, high_threshold)

def detect_horizontal_lines(contours, tolerance=10):
    horizontal_lines = []
    for contour in contours:
        for i in range(len(contour)):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i+1) % len(contour)][0]
            if abs(y1 - y2) < tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
    return horizontal_lines

def filter_edges_by_depth(image_lines, depth_lines, tolerance=10):
    filtered_lines = []
    for img_line in image_lines:
        for dpt_line in depth_lines:
            if abs(img_line[1] - dpt_line[1]) < tolerance:
                filtered_lines.append(img_line)
                break
    return filtered_lines

def main(image_path, depth_image_path):
    # 이미지 및 depth 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_image is None:
        print("이미지를 읽는 데 문제가 있습니다.")
        return

    # Edge Detection
    edges = detect_edges(image)
    depth_edges = detect_edges(depth_image)

    # 컨투어 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    depth_contours, _ = cv2.findContours(depth_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        vertices = len(approx)
        contour_area = cv2.contourArea(c)

        # Find bounding box and calculate center
        x, y, w, h = cv2.boundingRect(approx)
        center_x = x + w // 2
        center_y = y + h // 2

        if vertices == 4 and contour_area > 300:
            filtered_contours.append(approx)

    # 수평 에지 검출
    image_lines = detect_horizontal_lines(filtered_contours)
    depth_lines = detect_horizontal_lines(depth_contours)

    # Depth 에지를 사용하여 이미지 에지 필터링
    filtered_lines = filter_edges_by_depth(image_lines, depth_lines)

    # 필터링된 에지를 원본 이미지에 그리기
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Matplotlib을 사용하여 결과 시각화
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(depth_image, cmap='gray')
    axs[0, 1].set_title('Depth Image')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Canny Edges')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(result_image)
    axs[1, 1].set_title('Filtered Lines')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = './images/stair2.jpg'
    depth_image_path = './images/stair2_depth.png'
    main(image_path, depth_image_path)
