import pydicom
from PIL import Image

# 读取 dcm 文件
ds = pydicom.dcmread("trial/teeth.dcm")

# 获取像素数组并转换为图像
image = ds.pixel_array
im = Image.fromarray(image)

# 保存为常见格式
im.save("trial/teeth.png")
# 或者 im.save("output.jpg")
