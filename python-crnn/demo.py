import torch
from lib import dataset, utils
from PIL import Image
from Chinese_alphabet import alphabet
import lib.models.crnn as crnn

model_path = './data/netCRNN.pth'  # 模型权重路径
img_path = './data/title3.jpg'  # 测试图片路径
# 英文
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)  # 创建模型

# 选择设备
if torch.cuda.is_available():
    model = model.cuda()

# 加载模型
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

# 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列，使用英文字典时忽略大小写
converter = utils.strLabelConverter(alphabet, ignore_case=False)

# 图像大小转换器
transformer = dataset.resizeNormalize((100, 32))

# 读取并转换图像大小为100 x 32 w x h
image = Image.open(img_path).convert('L')
image = transformer(image)

if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)
print(image.size())

with torch.no_grad():
    model.eval()
    preds = model(image)  # (w c nclass) (26, 1, 37) 26为ctc生成路径长度也是传入rnn的时间步长，1是batchsize，37是字符类别数

_, preds = preds.max(2)  # 取可能性最大的indecis size (26, 1)
preds = preds.transpose(1, 0).contiguous().view(-1)  # 转成以为索引列表
# 转成字符序列
preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
