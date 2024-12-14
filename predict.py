import argparse
import pickle

import numpy as np
import torch

from model import TrajTransformer  # Đảm bảo file model.py nằm đúng thư mục
from utils import RnnParameterData


def load_model(model_path, parameters):
  """
  Tải mô hình đã huấn luyện từ file.
  :param model_path: Đường dẫn tới file mô hình đã lưu
  :param parameters: Các tham số để khởi tạo mô hình
  :return: Mô hình đã được tải
  """
  # Tạo đồ thị nếu cần
  g = gen_graph(parameters)

  # Khởi tạo mô hình
  model = TrajTransformer(parameters, graph=g)

  # Tải trọng số mô hình
  model.load_state_dict(torch.load(model_path))
  model.eval()  # Đặt mô hình ở chế độ đánh giá
  print('Model loaded successfully!')
  return model


model = load_model('D:/GCDAN-master/model.pth', parameters)
print('Model loaded successfully!')


def preprocess_data(data_path, max_len=20):
  """
  Tiền xử lý dữ liệu từ file dataset.
  :param data_path: Đường dẫn tới dataset pickle
  :param max_len: Độ dài tối đa của chuỗi nguồn
  :return: Dữ liệu đầu vào được xử lý
  """
  # Tải dữ liệu từ file pickle
  with open(data_path, 'rb') as f:
    data = pickle.load(f)

  # Khởi tạo các danh sách để chứa dữ liệu đã xử lý
  src_loc = []
  src_st = []
  src_ed = []

  # Lặp qua từng session trong dataset
  for user_id, user_data in data['data_filter'].items():
    sessions = user_data['raw_sessions']

    # Chỉ lấy đến độ dài tối đa (max_len)
    if len(sessions) > max_len:
      sessions = sessions[:max_len]

    user_src_loc = []
    user_src_st = []
    user_src_ed = []

    # Tiền xử lý từng session
    for session in sessions:
      session_id = session[0]  # ID session
      session[1]  # Thời gian của session (có thể không cần thiết nhưng có thể dùng nếu cần)

      # Ở đây ta có thể tính toán thời gian hoặc các thông số khác nếu cần.
      user_src_loc.append(session_id)  # Dùng session_id làm ví dụ
      user_src_st.append(0)  # Placeholder cho giá trị start, có thể thay đổi tùy nhu cầu
      user_src_ed.append(1)  # Placeholder cho giá trị end, có thể thay đổi tùy nhu cầu

    # Lưu dữ liệu đã tiền xử lý vào các danh sách chung
    src_loc.append(user_src_loc)
    src_st.append(user_src_st)
    src_ed.append(user_src_ed)

  # Padding nếu cần thiết
  src_loc = np.array(src_loc)
  src_st = np.array(src_st)
  src_ed = np.array(src_ed)

  if len(src_loc) < max_len:
    pad_length = max_len - len(src_loc)
    src_loc = np.pad(src_loc, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    src_st = np.pad(src_st, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    src_ed = np.pad(src_ed, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

  return {'src_loc': torch.LongTensor(src_loc), 'src_st': torch.LongTensor(src_st), 'src_ed': torch.LongTensor(src_ed)}


def predict(model, data):
  """
  Thực hiện dự đoán với mô hình và dữ liệu đầu vào.
  :param model: Mô hình đã được tải
  :param data: Dữ liệu đầu vào (tensor)
  :return: Kết quả dự đoán
  """
  # Lấy tensor từ dữ liệu đầu vào
  src_loc = data['src_loc']
  src_st = data['src_st']
  src_ed = data['src_ed']

  # Thực hiện dự đoán
  with torch.no_grad():
    output = model(src_loc, src_st, src_ed, src_loc, src_st, src_ed, src_loc.size(1), 0)

  # Lấy chỉ số dự đoán
  predicted = torch.argmax(output, dim=-1)
  return predicted


if __name__ == '__main__':
  # Thêm các tham số dòng lệnh
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True, help='Đường dẫn tới mô hình đã lưu (file .pth)')
  parser.add_argument('--data', type=str, required=True, help='Đường dẫn tới file dataset pickle')
  parser.add_argument('--loc_emb_size', type=int, default=512, help='Kích thước embedding cho vị trí')
  parser.add_argument('--uid_emb_size', type=int, default=128, help='Kích thước embedding cho user ID')
  parser.add_argument('--tim_emb_size', type=int, default=16, help='Kích thước embedding cho thời gian')
  parser.add_argument('--dropout_p', type=float, default=0.1, help='Tỉ lệ dropout')
  parser.add_argument('--max_len', type=int, default=20, help='Độ dài tối đa của chuỗi nguồn')
  args = parser.parse_args()

  # Tạo tham số cho mô hình
  parameters = RnnParameterData(
    loc_emb_size=args.loc_emb_size,
    uid_emb_size=args.uid_emb_size,
    tim_emb_size=args.tim_emb_size,
    dropout_p=args.dropout_p,
  )

  # Tải mô hình đã huấn luyện
  model = load_model('D:/GCDAN-master/model.pth', parameters)

  # Xử lý dữ liệu đầu vào
  data = preprocess_data('D:/GCDAN-master/data/foursquare.pk', max_len=args.max_len)

  # Thực hiện dự đoán
  result = predict(model, data)
  print('Prediction result:', result)
