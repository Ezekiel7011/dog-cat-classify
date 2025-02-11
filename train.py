import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from coatnet_pytorch_master.coatnet import *
from sklearn.metrics import confusion_matrix


logging.basicConfig(filename="training.log", level=logging.INFO)


def load_config(file_path):
    config = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() != "":
                key, value = line.strip().split("=")
                key = key.strip()
                value = value.strip().strip('"')
                config[key] = value
    return config


def main():
    try:
        config = load_config("config.txt")
        BATCH_SIZE = int(config.get("BATCH_SIZE", 64))
        NUM_EPOCHS = int(config.get("NUM_EPOCHS", 8))
        NUM_CLASSES = int(config.get("NUM_CLASSES", 1000))
        LEARNING_RATE = float(config.get("LEARNING_RATE", 0.00125))
        MOMENTUM = float(config.get("MOMENTUM", 0.9))
        WEIGHT_DECAY = float(config.get("WEIGHT_DECAY", 0.0005))
        TRAIN_DATA_PATH = config.get("TRAIN_DATA_PATH", "")
        VAL_DATA_PATH = config.get("VAL_DATA_PATH", "")
        WEIGHT_PATH = config.get("WEIGHT_PATH", "")
        IMG_SIZE = config.get("IMG_SIZE", "(224, 224)")
        IMG_SIZE = tuple(map(int, eval(IMG_SIZE)))

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        net = coatnet_0(num_classes=NUM_CLASSES)
        net.to(DEVICE)

        train_iter, test_iter = load_dataset(
            BATCH_SIZE, TRAIN_DATA_PATH, VAL_DATA_PATH, IMG_SIZE
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            net.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        record_train, record_test = train(
            net,
            train_iter,
            criterion,
            optimizer,
            NUM_EPOCHS,
            WEIGHT_PATH,
            DEVICE,
            lr_scheduler,
            test_iter,
        )

        learning_curve(record_train, record_test)
    except Exception as e:
        logging.exception("Exception occurred")


def load_dataset(batch_size, train_data_path, val_data_path, image_size):
    logging.info("Loading datasets...")

    # 定義轉換
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
            transforms.RandomCrop(image_size, padding=10),  # 隨機裁切
            transforms.ToTensor(),
        ]
    )

    # 載入訓練集
    train_set = torchvision.datasets.ImageFolder(train_data_path, transform=transform)

    # 載入驗證集
    test_set = torchvision.datasets.ImageFolder(val_data_path, transform=transform)

    # 創建資料載入器
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=1
    )

    logging.info("Datasets loaded successfully.")

    return train_iter, test_iter


def train(
    net,
    train_iter,
    criterion,
    optimizer,
    num_epochs,
    weight_path,
    device,
    lr_scheduler=None,
    test_iter=None,
):
    logging.info("Training started...")
    net.train()
    record_train = []
    record_test = []
    num_print = len(train_iter) // 4
    early_stop_counter = 0  # 初始化計數器

    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        logging.info("Epoch: %d/%d", epoch + 1, num_epochs)
        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in tqdm(enumerate(train_iter)):
            X, y = X.to(device), y.to(device)
            output = net(X)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            train_acc = 100.0 * correct / total

            if (i + 1) % num_print == 0:
                logging.info(
                    "Step: [%d/%d], Train Loss: %.3f, Train Acc: %.3f",
                    i + 1,
                    len(train_iter),
                    train_loss / (i + 1),
                    train_acc,
                )

        if lr_scheduler is not None:
            lr_scheduler.step()

        logging.info("Epoch %d completed. Time: %.2fs", epoch + 1, time.time() - start)

        if test_iter is not None:
            test_loss = test(net, test_iter, criterion, device)

            if test_loss > best_test_loss:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
                best_test_loss = test_loss

            if early_stop_counter >= 2:
                logging.info("Early stopping triggered. Training terminated.")
                # break

            record_test.append(test_loss)

        record_train.append(train_acc)
        torch.save(
            net.state_dict(), f"{weight_path}_{epoch + 1}_acc={record_test[epoch]}.pt"
        )
        logging.info("Weights saved for epoch %d", epoch + 1)

    torch.save(net.state_dict(), f"{weight_path}_full.pt")
    logging.info("Training completed.")
    return record_train, record_test

def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        logging.info("Testing...")
        for X, y in tqdm(test_iter):
            X, y = X.to(device), y.to(device)
            output = net(X)
            _, preds = torch.max(output, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_acc = correct / total * 100
    logging.info("Test Accuracy: %.2f%%", test_acc)
    net.train()

    # 顯示混淆矩陣
    confusion_mat = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)

    return test_acc


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")
    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("Learning Curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
