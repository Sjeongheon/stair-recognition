import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(img):
    low1, high1 = 5, 10
    edge1 = cv2.Canny(img, low1, high1)
    low2, high2 = 70, 140
    edge2 = cv2.Canny(img, low2, high2)
    low3, high3 = 100, 230
    edge3 = cv2.Canny(img, 100, 230)

    plt.subplot(221)
    plt.title('original')
    plt.imshow(img, cmap ='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('low : '+ str(low1) + ', high : ' + str(high1))
    plt.imshow(edge1, cmap ='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title('low : '+ str(low2) + ', high : ' + str(high2))
    plt.imshow(edge2, cmap ='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('low : '+ str(low3) + ', high : ' + str(high3))
    plt.imshow(edge3, cmap ='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = cv2.imread('images/stair2.jpg', cv2.IMREAD_GRAYSCALE)
    canny(image)

    depth = cv2.imread('images/stair2_depth.png', cv2.IMREAD_GRAYSCALE)
    canny(depth)
