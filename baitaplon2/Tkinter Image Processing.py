import cv2
import numpy as np
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

# Hàm chọn và mở ảnh
def open_image():
    global img, file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)
        show_image(img)

# Hiển thị ảnh trong giao diện Tkinter
def show_image(cv_img):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv_img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Hàm xử lý Watershed
def watershed():
    global img
    if img is None:
        print("Vui lòng chọn một ảnh trước!")
        return

    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển sang nhị phân
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tìm vùng chắc chắn nền và đối tượng
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Tạo markers
    unknown = cv2.subtract(sure_bg.astype(np.uint8), sure_fg.astype(np.uint8))
    _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers = markers + 1
    markers[unknown == 255] = 0

    # Áp dụng Watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]  # Đường viền màu đỏ

    # Hiển thị kết quả
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Result')
    plt.axis('off')
    plt.show()

# Hàm xử lý GrabCut
def grabcut():
    global img
    if img is None:
        print("Vui lòng chọn một ảnh trước!")
        return

    original_image = img.copy()
    height, width = original_image.shape[:2]

    # Chọn ROI (Region of Interest)
    print("Vui lòng chọn vùng bao quanh đối tượng...")
    roi = cv2.selectROI("Chọn vùng đối tượng", original_image, showCrosshair=True, fromCenter=False)
    if roi == (0, 0, 0, 0):
        print("Vùng chọn không hợp lệ!")
        cv2.destroyAllWindows()
        return

    # Lấy tọa độ ROI
    x, y, w, h = roi

    # Khởi tạo các mô hình nền và đối tượng
    mask = np.zeros((height, width), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Áp dụng GrabCut với ROI
    rect = (x, y, w, h)
    try:
        cv2.grabCut(original_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print("Lỗi trong GrabCut:", e)
        cv2.destroyAllWindows()
        return

    # Tạo mask cuối cùng
    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Áp dụng mask lên ảnh gốc
    result = cv2.bitwise_and(original_image, original_image, mask=final_mask)

    # Cắt ảnh theo vùng chọn
    cropped_result = result[y:y + h, x:x + w]

    # Hiển thị kết quả
    cv2.imshow("Kết quả GrabCut", result)
    cv2.imshow("Ảnh cắt theo ROI", cropped_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Giao diện Tkinter
root = Tk()
root.title("Watershed & GrabCut")

select_button = Button(root, text="Chọn Ảnh", command=open_image)
select_button.pack()

watershed_button = Button(root, text="Watershed", command=watershed)
watershed_button.pack()

grabcut_button = Button(root, text="GrabCut", command=grabcut)
grabcut_button.pack()

img_label = Label(root)
img_label.pack()

img = None
file_path = ""

root.mainloop()
