import numpy as np
import matplotlib.pyplot as plt

def imgShow(img):
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)


def transparencyOverlay(pixel_color, color, alpha):
    pixel_color = alpha * color + (1 - alpha) * pixel_color
    return pixel_color


def addRectangle(img, x1, y1, x2, y2, color, alpha):
    '''
    添加一个带透明度的矩阵框
    para x1: 矩阵框左上角的纵坐标(上边为0)
    para y1: 矩阵框左上角的横坐标(左边为0)
    para x2: 矩阵框右下角的纵坐标(上边为0)
    para y2: 矩阵框右下角的横坐标(左边为0)
    para color: 3为数组，[R, G, B]
    para alpha: 透明度，alpha=1完全不透明，alpha=0完全透明
    return 添加矩阵边框后的图像
    '''
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(img.shape[1] - 1, x2))
    y2 = int(min(img.shape[0] - 1, y2))
    alpha = float(alpha)
    img = img.copy()
    color = np.array(color)
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            img[y, x] = transparencyOverlay(img[y, x], color, alpha)
    return img


def addRectangleLine(img, x1, y1, x2, y2, color, alpha, line_width=1):
    '''
    添加一个带透明度的矩阵边框
    para x1: 矩阵边框左上角的纵坐标(上边为0)
    para y1: 矩阵边框左上角的横坐标(左边为0)
    para x2: 矩阵边框右下角的纵坐标(上边为0)
    para y2: 矩阵边框右下角的横坐标(左边为0)
    para color: 3为数组，[R, G, B]
    para alpha: 透明度，alpha=1完全不透明，alpha=0完全透明
    para line_width: 矩阵边框线宽
    return 添加矩阵边框后的图像
    '''
    assert 0 <= alpha, 'alpha必须大于等于0，小于等于1'
    assert alpha <= 1, 'alpha必须大于等于0，小于等于1'
    img = img.copy()
    color = np.array(color)
    line_x = -line_width // 2
    line_y = line_x + line_width
    img = addRectangle(img, x1 + line_x, y1, x1 + line_y, y2, color, alpha)
    img = addRectangle(img, x2 + line_x, y1, x2 + line_y, y2, color, alpha)
    img = addRectangle(img, x1, y1 + line_x, x2, y1 + line_y, color, alpha)
    img = addRectangle(img, x1, y2 + line_x, x2, y2 + line_y, color, alpha)
    return img
