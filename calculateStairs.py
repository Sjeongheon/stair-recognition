import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(img_path):
    # 이미지 읽기
    img = cv2.imread(img_path)

    # Gaussian Blur 적용
    img_blur = cv2.GaussianBlur(img, (3, 7), 0)

    # Canny Edge Detection 적용
    dst = cv2.Canny(img_blur, 80, 160, apertureSize=3)
    out_img = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    control = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    y_keeper_for_lines = []
    lines = cv2.HoughLinesP(dst, 1, np.pi/180, 30, minLineLength=40, maxLineGap=5)

    if lines is not None:
        for i in range(1, len(lines)):
            l = lines[i][0]
            cv2.line(control, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

        l = lines[0][0]
        cv2.line(out_img, (0, l[1]), (img.shape[1], l[1]), (0, 0, 255), 3, cv2.LINE_AA)
        y_keeper_for_lines.append(l[1])

        okey = True
        stair_counter = 1

        for i in range(1, len(lines)):
            l = lines[i][0]
            for m in y_keeper_for_lines:
                if abs(m - l[1]) < 20:
                    okey = False
                    break

            if okey:
                cv2.line(out_img, (0, l[1]), (img.shape[1], l[1]), (0, 0, 255), 3, cv2.LINE_AA)
                y_keeper_for_lines.append(l[1])
                stair_counter += 1

            okey = True

        cv2.putText(out_img, "Stair number: " + str(stair_counter), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(img)
    axs[0].set_title('Source')
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(control, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Control')
    axs[1].axis('off')

    axs[2].imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
    axs[2].set_title('Detected Lines')
    axs[2].axis('off')

    axs[3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[3].set_title('Before')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main('images/stair3.jpg')
