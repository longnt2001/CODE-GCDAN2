import matplotlib.pyplot as plt


# Hiển thị trực quan hành trình trong phiên đầu tiên
def visualize_session(session):
  locations = [loc[0] for loc in session]  # Lấy danh sách các địa điểm
  timestamps = [loc[1] for loc in session]  # Lấy danh sách thời gian

  # Tạo sơ đồ
  plt.figure(figsize=(10, 6))
  plt.plot(locations, marker='o', label='Hành trình')
  for i, (loc, time) in enumerate(zip(locations, timestamps)):
    plt.text(i, loc, time, fontsize=8, ha='right')

  plt.title('Hành trình phiên đầu tiên (Raw Session)')
  plt.xlabel('Thứ tự điểm đến')
  plt.ylabel('ID Địa điểm')
  plt.legend()
  plt.grid()
  plt.show()


# Visualize phiên đầu tiên của người dùng
if 'raw_sessions' in first_user_data and first_user_data['raw_sessions']:
  print('Hành trình trong phiên đầu tiên:')
  print(first_user_data['raw_sessions'][0])
  visualize_session(first_user_data['raw_sessions'][0])
else:
  print('Người dùng không có dữ liệu phiên (sessions).')
