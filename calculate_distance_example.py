import cv2
import numpy as np
import matplotlib.pyplot as plt

def measure_distance(depth_image, point1, point2):
    # 두 점의 깊이 값을 추출
    depth1 = float(depth_image[point1[1], point1[0]])
    depth2 = float(depth_image[point2[1], point2[0]])
    
    # 두 점 간의 유클리드 거리 계산
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # 픽셀 간의 거리와 깊이 값을 이용한 실제 거리 계산
    actual_distance = np.sqrt(distance**2 + (depth2 - depth1)**2)
    
    return actual_distance

def main(image_path, depth_image_path, point1, point2):
    # 이미지 및 depth 이미지 읽기
    image = cv2.imread(image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_image is None:
        print("이미지를 읽는 데 문제가 있습니다.")
        return

    # 거리를 측정할 두 점
    point1 = (int(point1[0]), int(point1[1]))
    point2 = (int(point2[0]), int(point2[1]))

    # 두 점 간의 거리 계산
    distance = measure_distance(depth_image, point1, point2)
    
    # 결과 시각화
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.circle(image_rgb, point1, 5, (255, 0, 0), -1)
    cv2.circle(image_rgb, point2, 5, (0, 255, 0), -1)
    cv2.line(image_rgb, point1, point2, (0, 0, 255), 2)
    plt.imshow(image_rgb)
    plt.title(f'Distance: {distance:.2f} units')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 이미지 경로 설정
    dir_path = './Stair dataset with depth maps/data/train/'
    image_path = dir_path + 'images/color_10_night_l.jpg'
    depth_image_path = dir_path + 'depthes/color_10_night_l.png'
    
    # 거리를 측정할 두 점 설정 (여기서 (x, y) 좌표를 지정)
    point1 = (500, 450)
    point2 = (500, 500)
    
    main(image_path, depth_image_path, point1, point2)
