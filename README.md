# Ứng Dụng Phân Tích Cảm Xúc Tiếng Việt Sử Dụng PhoBERT

## 📌 Tổng Quan
Phân tích cảm xúc văn bản tiếng Việt dựa trên mô hình PhoBERT được fine-tune. Mô hình được huấn luyện trên dataset UIT-VSFC để phân loại cảm xúc thành 3 loại: **Tiêu cực**, **Trung lập** và **Tích cực**.

## 📂 Cấu Trúc Dự Án
```plaintext
Project/
├── app.py               # Ứng dụng Flask chính
├── templates/           # Thư mục template HTML
│   └── index.html       # Giao diện người dùng
├── result/
|   ├──checkpoint/
|   ├──final_model/        # Mô hình đã huấn luyện
├── SAV.ipynb              # Notebook huấn luyện mô hình
└── VSFC/                  # Dataset
    ├── train/
    │   ├── sents.txt       # Dữ liệu văn bản huấn luyện
    │   ├── sentiments.txt  # Nhãn cảm xúc
    │   └── topics.txt      # Chủ đề
    └── test/
    │   ├── sents.txt          # Dữ liệu kiểm tra\
    │   ├── sentiments.txt
    │   └── topics.txt 
    ...
    ```
## ⚙️ Yêu Cầu Hệ Thống
### Phần mềm
```plaintext
# requirements.txt
flask==2.0.3
torch==2.4.1+cu121
transformers==4.44.2
underthesea==6.8.4
pandas==2.0.3
scikit-learn==1.2.2
nltk==3.8.1
python-crfsuite==0.9.11
```
### Phần cứng
RAM tối thiểu: 8GB (16GB khuyến nghị cho huấn luyện).
GPU: NVIDIA với CUDA hỗ trợ (khuyến nghị cho huấn luyện).
## ✍️📜 Cách Sử Dụng
### Khởi chạy ứng dụng:
``` bash
python app.py
```
### Truy cập giao diện web tại:

```arduino
http://localhost:5000
```
### Nhập văn bản và nhận kết quả:

Ví dụ đầu vào:
```plaintext
"Giảng viên nhiệt tình, giáo trình rõ ràng dễ hiểu"
```
Kết quả:
```plaintext
Tích cực ✅
```
## 🤖🖥️ Huấn Luyện Mô Hình
### Quy trình được thực hiện trong file train_model.ipynb:

1. Tiền xử lý dữ liệu
Tách từ với Underthesea.
Chuẩn hóa định dạng văn bản.
Chia tập train/test tỉ lệ 80/20.
2. Cấu hình mô hình
Kiến trúc: PhoBERT-base.
Số lớp: 3 (Tiêu cực/Trung lập/Tích cực).
Độ dài tối đa văn bản: 128 tokens.
3. Tham số huấn luyện
Số epoch: 5.
Batch size: 16.
Tốc độ học: 5e-5.
Độ chính xác cuối cùng: 88.95%.
4. Kết quả đánh giá
```plaintext
Accuracy: 0.8894504106127605
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.94      0.91      1409
           1       0.43      0.26      0.32       167
           2       0.93      0.91      0.92      1590

    accuracy                           0.89      3166
   macro avg       0.74      0.70      0.72      3166
weighted avg       0.88      0.89      0.88      3166
```
