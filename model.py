#导入相关包
from torch import nn



"""定义生成器网络结构"""
class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.ReLU(inplace=True), bn=True):
        seq = []
        seq += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn is True:
          seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(20, 64*8, stride=1, padding=0)]
    seq += [CBA(64*8, 64*4)]
    seq += [CBA(64*4, 64*2)]
    seq += [CBA(64*2, 64)]
    seq += [CBA(64, 1, activation=nn.Tanh(), bn=False)]

    self.generator_network = nn.Sequential(*seq)

  def forward(self, z):
      out = self.generator_network(z)

      return out


"""定义判别器网络结构"""
class Discriminator(nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()

    def CBA(in_channel, out_channel, kernel_size=4, stride=2, padding=1, activation=nn.LeakyReLU(0.1, inplace=True)):
        seq = []
        seq += [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        seq += [nn.BatchNorm2d(out_channel)]
        seq += [activation]

        return nn.Sequential(*seq)

    seq = []
    seq += [CBA(1, 64)]
    seq += [CBA(64, 64*2)]
    seq += [CBA(64*2, 64*4)]
    seq += [CBA(64*4, 64*8)]
    self.feature_network = nn.Sequential(*seq)

    self.critic_network = nn.Conv2d(64*8, 1, kernel_size=4, stride=1)

  def forward(self, x):
      out = self.feature_network(x)

      feature = out
      feature = feature.view(feature.size(0), -1)

      out = self.critic_network(out)

      return out, feature
