# CoAtNet Cat and Dog Classification

這是一個使用 CoAtNet 對貓和狗進行分類的深度學習專案。本專案包含了一個預訓練的模型`coatnet_catdog_full.pt`，可供立即使用。

## 模型性能

模型在測試集上的表現如下：

- Precision (準確率): 0.86
- Recall (召回率): 0.85
- F1-Score: 0.85
- Accuracy (準確度): 0.85

混淆矩陣如下：

|            | 預測為 Cat | 預測為 Dog |
| ---------- | ---------- | ---------- |
| 實際為 Cat | 287        | 46         |
| 實際為 Dog | 51         | 282        |

## 安裝

首先，請確保安裝了 Python 3.6+。然後，安裝所需的依賴項目：

```
pip install -r requirements.txt
```

## 使用方法

為了使用預訓練的模型進行預測，請運行：

```
python test.py --config test_config.txt
```

## 訓練您自己的模型

如果您希望訓練自己的模型，請使用以下命令：

```
python train.py --config config.txt
```

## 參考

本專案的 CoAtNet 實現參考自[這裡](https://github.com/chinhsuanwu/coatnet-pytorch)。感謝原作者的貢獻。

## 資料

專案中包含`train`和`val`兩個資料夾，內含少量資料供展示和測試使用。
