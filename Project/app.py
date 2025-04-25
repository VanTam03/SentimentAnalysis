# Import các thư viện cần thiết
from flask import Flask, request, render_template, jsonify 
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Từ Hugging Face để tải mô hình phân loại và tokenizer
import torch  

# Khởi tạo ứng dụng Flask
app = Flask(__name__) 

# Tải mô hình và tokenizer
model_name = r"D:\HKII 2425\CNLTHD\Project\results\final_model"  # Đường dẫn đến thư mục chứa mô hình và tokenizer đã lưu
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Tải tokenizer từ đường dẫn đã cho
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Tải mô hình phân loại với 3 nhãn (Tiêu cực, Trung lập, Tích cực)
model.eval()  # Chuyển mô hình sang chế độ đánh giá (không huấn luyện), tắt các lớp như dropout để dự đoán chính xác

# Hàm dự đoán cảm xúc từ văn bản đầu vào
def predict_sentiment(text):
    if not text.strip():  # Kiểm tra nếu văn bản rỗng hoặc chỉ chứa khoảng trắng
        return "Vui lòng nhập văn bản."  # Trả về thông báo yêu cầu người dùng nhập văn bản
    try:
        # Tokenize văn bản: chuyển văn bản thành định dạng mà mô hình có thể hiểu (input_ids, attention_mask)
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        
        with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm tài nguyên (chỉ dự đoán, không huấn luyện)
            outputs = model(**inputs)  # Đưa inputs qua mô hình để nhận kết quả dự đoán (logits)
        
        # Lấy nhãn dự đoán: tìm lớp có xác suất cao nhất từ logits
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()  # torch.argmax tìm chỉ số của giá trị lớn nhất trong logits
        labels = ["Tiêu cực", "Trung lập", "Tích cực"]  # Danh sách ánh xạ từ chỉ số (0, 1, 2) sang nhãn cảm xúc
        return labels[predicted_class_id]  # Trả về nhãn cảm xúc tương ứng
    except Exception as e:  # Bắt lỗi nếu có vấn đề trong quá trình dự đoán
        return f"Lỗi: {str(e)}"  # Trả về thông báo lỗi với chi tiết

# Route chính của ứng dụng, hỗ trợ cả phương thức GET và POST
@app.route("/", methods=["GET", "POST"])  # Định nghĩa route "/" (trang chính), chấp nhận cả GET và POST
def index():
    if request.method == "POST":  # Nếu yêu cầu là POST (người dùng gửi dữ liệu qua form)
        # Lấy văn bản từ form trong yêu cầu POST
        text = request.form.get("text")  # Lấy giá trị của trường "text" từ form
        if not text:  # Kiểm tra nếu văn bản rỗng
            return jsonify({"error": "Vui lòng nhập văn bản"}), 400  # Trả về JSON với thông báo lỗi và mã trạng thái 400 (Bad Request)

        # Dự đoán cảm xúc từ văn bản
        sentiment = predict_sentiment(text)  # Gọi hàm predict_sentiment để dự đoán

        # Kiểm tra nếu yêu cầu là AJAX (gửi từ JavaScript)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"sentiment": sentiment})  # Trả về JSON chứa kết quả dự đoán cho yêu cầu AJAX
        else:
            # Nếu không phải AJAX, trả về trang HTML với kết quả
            return render_template("index.html", sentiment=sentiment)  # Render template index.html và truyền biến sentiment
    # Nếu yêu cầu là GET (truy cập trang lần đầu hoặc refresh)
    return render_template("index.html", sentiment=None)  # Render template index.html mà không có kết quả dự đoán
# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)  # Chạy ứng dụng ở chế độ debug (tự động reload khi thay đổi code, hiển thị lỗi chi tiết)