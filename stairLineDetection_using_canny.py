import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image, low_threshold=70, high_threshold=140):
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

def filter_edges_by_depth(image_lines, depth_lines, tolerance=10):
    filtered_lines = []
    for img_line in image_lines:
        for dpt_line in depth_lines:
            if abs(img_line[1] - dpt_line[1]) < tolerance:  # 수평 에지의 y값을 비교하여 유사한 에지를 필터링
                filtered_lines.append(img_line)
                break
    return filtered_lines
  

if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = './images/stair3.jpg'
    depth_image_path = './images/stair2_depth.png'
      # 이미지 및 depth 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_image is None:
        print("이미지를 읽는 데 문제가 있습니다.")

    # Edge Detection
    edges = detect_edges(image)
    depth_edges = detect_edges(depth_image)

    # 수평 에지 검출
    image_lines = detect_horizontal_lines(edges)
    depth_lines = detect_horizontal_lines(depth_edges)

    # Depth 에지를 사용하여 이미지 에지 필터링
    filtered_lines = filter_edges_by_depth(image_lines, depth_lines)

    # 필터링된 에지를 원본 이미지에 그리기
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 20)

    for x1, y1, x2, y2 in image_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    for x1, y1, x2, y2 in depth_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

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
