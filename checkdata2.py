import pickle

file_path = 'data/foursquare.pk'

with open(file_path, 'rb') as f:
  data_neural = pickle.load(f, encoding='latin1')

# Hiển thị toàn bộ nội dung của tệp
print('Nội dung của tệp:')
print(data_neural)
