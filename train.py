
#导入相关包
import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import *
from mnist_data import *


class image_data_set(Dataset):
    def __init__(self, data):
        self.images = data[:,:,:,None]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64, interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])




def main(args):
    # 指定设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # batch_size默认128
    batch_size = args.batch_size


    # 加载训练数据
    train_set = image_data_set(train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)






    # 加载模型
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 训练模式
    G.train()
    D.train()

    # 设置优化器
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    """
    训练
    """

    # 开始训练
    for epoch in range(args.epochs):
        # 定义初始损失
        log_g_loss, log_d_loss = 0.0, 0.0
        for images in train_loader:
            images = images.to(device)

            ## 训练判别器 Discriminator
            # 定义真标签（全1）和假标签（全0）   维度：（batch_size）
            label_real = torch.full((images.size(0),), 1.0).to(device)
            label_fake = torch.full((images.size(0),), 0.0).to(device)

            # 定义潜在变量z    维度：(batch_size,20,1,1)
            z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
            # 潜在变量喂入生成网络--->fake_images:(batch_size,1,64,61)
            fake_images = G(z)

            # 真图像和假图像送入判别网络，得到d_out_real、d_out_fake   维度：都为（batch_size,1,1,1）
            d_out_real, _ = D(images)
            d_out_fake, _ = D(fake_images)

            # 损失计算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # 误差反向传播，更新损失
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            ## 训练生成器 Generator
            # 定义潜在变量z    维度：(batch_size,20,1,1)
            z = torch.randn(images.size(0), 20).to(device).view(images.size(0), 20, 1, 1).to(device)
            fake_images = G(z)

            # 假图像喂入判别器，得到d_out_fake   维度：（batch_size,1,1,1）
            d_out_fake, _ = D(fake_images)

            # 损失计算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # 误差反向传播，更新损失
            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            ## 累计一个epoch的损失，判别器损失和生成器损失分别存放到log_d_loss、log_g_loss中
            log_d_loss += d_loss.item()
            log_g_loss += g_loss.item()

        ## 打印损失
        print(f'epoch {epoch}, D_Loss:{log_d_loss / 128:.4f}, G_Loss:{log_g_loss / 128:.4f}')




    ## 展示生成器存储的图片，存放在result文件夹下的G_out.jpg
    z = torch.randn(8, 20).to(device).view(8, 20, 1, 1).to(device)
    fake_images = G(z)
    torchvision.utils.save_image(fake_images,f"result/G_out.jpg")


    """
    测试
    """

    ## 定义缺陷计算的得分
    def anomaly_score(input_image, fake_image, D):
      # Residual loss 计算
      residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

      # Discrimination loss 计算
      _, real_feature = D(input_image)
      _, fake_feature = D(fake_image)
      discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

      # 结合Residual loss和Discrimination loss计算每张图像的损失
      total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
      # 计算总损失，即将一个batch的损失相加
      total_loss = total_loss_by_image.sum()

      return total_loss, total_loss_by_image, residual_loss



    # 加载测试数据
    test_set = image_data_set(test)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False)
    input_images = next(iter(test_loader)).to(device)

    # 定义潜在变量z  维度：（5，20，1，1）
    z = torch.randn(5, 20).to(device).view(5, 20, 1, 1)
    # z的requires_grad参数设置成Ture,让z可以更新
    z.requires_grad = True

    # 定义优化器
    z_optimizer = torch.optim.Adam([z], lr=1e-3)

    # 搜索z
    for epoch in range(5000):
      fake_images = G(z)
      loss, _, _ = anomaly_score(input_images, fake_images, D)

      z_optimizer.zero_grad()
      loss.backward()
      z_optimizer.step()

      if epoch % 1000 == 0:
        print(f'epoch: {epoch}, loss: {loss:.0f}')




    fake_images = G(z)

    _, total_loss_by_image, _ = anomaly_score(input_images, fake_images, D)

    print(total_loss_by_image.cpu().detach().numpy())

    torchvision.utils.save_image(input_images, f"result/Nomal.jpg")
    torchvision.utils.save_image(fake_images, f"result/ANomal.jpg")



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
