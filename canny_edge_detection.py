import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(img):
    edge1 = cv2.Canny(img, 30, 200)
    edge2 = cv2.Canny(img, 70, 200)
    edge3 = cv2.Canny(img, 100, 230)

    plt.subplot(221)
    plt.title('original')
    plt.imshow(img, cmap ='gray')
    plt.axis('off')

    plt.subplot(222)
    plt.title('low : 50, high : 200')
    plt.imshow(edge1, cmap ='gray')
    plt.axis('off')

    plt.subplot(223)
    plt.title('low : 100, high : 200')
    plt.imshow(edge2, cmap ='gray')
    plt.axis('off')

    plt.subplot(224)
    plt.title('low : 170, high : 230')
    plt.imshow(edge3, cmap ='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = cv2.imread('images/stair.jpg', cv2.IMREAD_GRAYSCALE)
    canny(image)