import cv2
import numpy as np
import matplotlib.pyplot as plt

def adaptive_threshold(image):
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10) # 조정된 매개변수: blockSize=15, C=10
    return adaptive_thresh

def detect_edges(image):
    edges = cv2.Canny(image, 80, 200)
    return edges

def detect_horizontal_lines(contours, tolerance=10):
    horizontal_lines = []
    for contour in contours:
        for i in range(len(contour)):
            x1, y1 = contour[i][0]
            x2, y2 = contour[(i+1) % len(contour)][0]
            if abs(y1 - y2) < tolerance:
                horizontal_lines.append((x1, y1, x2, y2))
    return horizontal_lines

def filter_wide_contours(contours, min_width=50):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > h and w > min_width:  # 가로가 세로보다 길고, 최소 너비 조건을 만족하는 경우만 필터링
            filtered_contours.append(contour)
    return filtered_contours

def filter_edges_by_depth(image_lines, depth_lines, tolerance=10):
    filtered_lines = []
    for img_line in image_lines:
        for dpt_line in depth_lines:
            if abs(img_line[1] - dpt_line[1]) < tolerance:
                filtered_lines.append(img_line)
                break
    return filtered_lines

def enhance_gradient(image):
    # Sobel 필터를 사용하여 x축과 y축 그라디언트를 계산
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # 그라디언트의 절대값을 구하고 합성
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # 그라디언트를 더 강조
    enhanced_grad = cv2.convertScaleAbs(grad, alpha=2, beta=0)  # alpha 값으로 강도를 조절
    
    return enhanced_grad

def main(image_path, depth_image_path):
    # 이미지 및 depth 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_image is None:
        print("이미지를 읽는 데 문제가 있습니다.")
        return

    # 가장자리 마스크 적용
    mask = np.ones(image.shape, dtype=np.uint8) * 255
    mask[0:10, :] = 0  # 상단 가장자리 마스크
    mask[-10:, :] = 0  # 하단 가장자리 마스크
    mask[:, 0:10] = 0  # 왼쪽 가장자리 마스크
    mask[:, -10:] = 0  # 오른쪽 가장자리 마스크
    masked_image = cv2.bitwise_and(image, mask)

    # Gradient를 강화한 depth 이미지
    enhanced_depth = enhance_gradient(depth_image)

    # Adaptive Thresholding
    adaptive_thresh_image = adaptive_threshold(masked_image)
    adaptive_thresh_depth = adaptive_threshold(enhanced_depth)

    # Edge Detection using Canny
    edges = detect_edges(adaptive_thresh_image)
    depth_edges = detect_edges(adaptive_thresh_depth)

    # Find Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    depth_contours, _ = cv2.findContours(depth_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter Wide Contours
    filtered_contours = filter_wide_contours(contours)
    filtered_depth_contours = filter_wide_contours(depth_contours)

    # Detect Horizontal Lines
    image_lines = detect_horizontal_lines(filtered_contours)
    depth_lines = detect_horizontal_lines(filtered_depth_contours)

    # Filter Lines by Depth
    filtered_lines = filter_edges_by_depth(image_lines, depth_lines)

    # Draw filtered lines on the original image
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contour_image = np.copy(result_image)
    
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw Contours
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 0, 255), 2)

    # Display results
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(depth_image, cmap='gray')
    axs[0, 1].set_title('Depth Image')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(enhanced_depth, cmap='gray')
    axs[0, 2].set_title('Enhanced Depth Gradient')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Canny Edges')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(result_image)
    axs[1, 1].set_title('Filtered Lines')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(contour_image)
    axs[1, 2].set_title('Contours on Image')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 이미지 경로 설정
    dir_path = './Stair dataset with depth maps/data/train/'
    image_path = dir_path + 'images/color_10_night_l.jpg'
    depth_image_path = dir_path + 'depthes/color_10_night_l.png'
    main(image_path, depth_image_path)
