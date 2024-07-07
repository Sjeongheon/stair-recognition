import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image, low_threshold=9, high_threshold=240):
    blur = cv2.GaussianBlur(image, (5,5), 0)
    return cv2.Canny(blur, low_threshold, high_threshold)

def detect_horizontal_lines(edges, rho=1, theta=np.pi/180, threshold=30, min_line_length=40, max_line_gap=5):
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    horizontal_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y1 - y2) < 10:  # 수평선을 검출하기 위해 y값 차이가 작은 선만 선택
                horizontal_lines.append((x1, y1, x2, y2))
    return horizontal_lines

def find_flat_areas(depth_image, threshold=10):
    depth_diff = cv2.Laplacian(depth_image, cv2.CV_64F)
    flat_areas = np.where(np.abs(depth_diff) < threshold, 255, 0).astype(np.uint8)
    return flat_areas

def filter_lines_by_flat_areas(lines, flat_areas):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if flat_areas[y1, x1] == 255 and flat_areas[y2, x2] == 255:
            filtered_lines.append(line)
    return filtered_lines

if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = './images/stair2.jpg'
    depth_image_path = './images/stair2_depth.png'

    # 이미지 및 depth 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_image is None:
        print("이미지를 읽는 데 문제가 있습니다.")
        exit()

    # Edge Detection
    edges = detect_edges(image)

    # 수평 에지 검출
    image_lines = detect_horizontal_lines(edges)

    # Depth 이미지에서 평평한 영역 검출
    flat_areas = find_flat_areas(depth_image)

    # 평평한 영역을 사용하여 수평 에지 필터링
    filtered_lines = filter_lines_by_flat_areas(image_lines, flat_areas)

    # 필터링된 에지를 원본 이미지에 그리기
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Matplotlib을 사용하여 결과 시각화
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(depth_image, cmap='gray')
    axs[0, 1].set_title('Depth Image')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(flat_areas, cmap='gray')
    axs[0, 2].set_title('Flat Areas in Depth Image')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Canny Edges')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(result_image)
    axs[1, 1].set_title('Filtered Lines')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()
