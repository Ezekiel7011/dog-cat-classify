import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from coatnet_pytorch_master.coatnet import *

# 讀取配置文件
config_path = "./test_config.txt"
config = {}
with open(config_path, "r") as f:
    for line in f:
        key, value = line.strip().split("=")
        config[key] = value

# 解析類別列表和預訓練模型權重路徑
classes = config["classes"].split(",")
checkpoint_path = config["checkpoint"]

# 創建一個與預訓練模型相同結構的模型
model = coatnet_0(num_classes=len(classes))

# 載入預訓練模型權重，並將其應用於新的目標模型
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint, strict=False)

# 確認模型已經成功載入權重
print("Model loaded successfully with pretrained weights.")

# 定義圖片預處理的函式
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# 定義預測函式
def predict(image_path):
    image = Image.open(image_path)
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 維度
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    predicted_label = classes[predicted_idx.item()]
    print("output:", outputs)
    print("predicted_idx:", predicted_idx.item())
    return predicted_label


# 定義選擇圖片並顯示預測結果的函式
def choose_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # 預測標籤
        predicted_label = predict(file_path)
        result_label.config(text=f"Predicted Label: {predicted_label}")

        # 加載圖片並縮放
        image = Image.open(file_path)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 300  # 目標寬度
        new_height = int(new_width / aspect_ratio)  # 等比例縮放計算高度
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

        # 顯示縮放後的圖片
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


# 建立主視窗
root = tk.Tk()
root.title("辨識應用程式")

# 添加選擇圖片按鈕
choose_button = tk.Button(root, text="選擇圖片", command=choose_image)
choose_button.pack()

# 添加圖片顯示區域
image_label = tk.Label(root)
image_label.pack()

# 添加顯示預測結果的標籤
result_label = tk.Label(root, text="Predicted Label: None")
result_label.pack()

# 開始執行主迴圈
root.mainloop()
