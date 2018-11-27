from torchvision import transforms
from run_code import run_style_transfer
from load_img import load_img, show_img
from torch.autograd import Variable

style_img = load_img('./picture/style.png')
style_img = Variable(style_img)
content_img = load_img('./picture/cat.jpg')
content_img = Variable(content_img)

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)

show_img(out)
save_pic = transforms.ToPILImage()(out.squeeze(0))
save_pic.save('./picture/saved_picture.png')