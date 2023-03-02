import cv2 
# 使用差值来提高图片的分辨率
img = cv2.imread('/Users/jachin/Downloads/IMG_2567.JPG')
width = img.shape[0]
height = img.shape[1]
r = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('/Users/jachin/Downloads/tt.jpg', r)
print('ss')