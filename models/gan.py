import torch.nn as nn

DECODER_IN = [100, 64, 64, 256, 128, 128, 64, 64, 64, 64]
DECODER_OUT = [64, 64, 256, 128, 128, 64, 64, 64, 64, 1]
DECODER_KERNEL = [8, 3, 3, 3, 3, 3, 3, 3, 3, 3]
DECODER_STRIDE = [1, 1, 1, 2, 1, 2, 1, 2, 2, 2]
DECODER_PADDING = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
DECODER_OUT_PADDING = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
ENCODER_IN = [1, 64, 64, 64, 64, 128, 128, 256, 128, 64]
ENCODER_OUT = [64, 64, 64, 64, 128, 128, 256, 128, 64, 1]
ENCODER_KERNEL = [4, 4, 4, 3, 4, 3, 4, 3, 3, 8]
ENCODER_STRIDE = [2, 2, 2, 1, 2, 1, 2, 1, 1, 1]
ENCODER_PADDING = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = self._make_deconv_layers(DECODER_IN, DECODER_OUT, DECODER_KERNEL, DECODER_STRIDE, DECODER_PADDING,
                                                DECODER_OUT_PADDING)
    def _make_deconv_layers(self, inplanes, outplanes, kernel, stride, padding, output_padding):
        modules = []
        for i in range(len(inplanes)):
          if i != len(inplanes)-1:
            modules.extend([
                nn.ConvTranspose2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                                    stride=stride[i], padding=padding[i], output_padding=output_padding[i], bias=True),
            nn.BatchNorm2d(outplanes[i]),
            nn.ReLU(True)])
          else:
              modules.extend([
                nn.ConvTranspose2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                                    stride=stride[i], padding=padding[i], output_padding=output_padding[i], bias=True),
                nn.Sigmoid()])
        return nn.Sequential(*modules)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = self._make_conv_layers(ENCODER_IN, ENCODER_OUT, ENCODER_KERNEL, ENCODER_STRIDE, ENCODER_PADDING)

    def _make_conv_layers(self, inplanes, outplanes, kernel, stride, padding):
        modules = []
        for i in range(len(inplanes)):
          if i != len(inplanes)-1:
            modules.extend([
                nn.Conv2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                          stride=stride[i],
                          padding=padding[i], bias=True),
                nn.BatchNorm2d(outplanes[i]),
                nn.LeakyReLU(0.2, inplace=True)])
          else:
              modules.extend([
                nn.Conv2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                          stride=stride[i],
                          padding=padding[i], bias=True),
                nn.Sigmoid()])
        return nn.Sequential(*modules)

    def forward(self, x):
        y = self.main(x)
        y = y.view(-1)
        return {'pred': y}


class GAN():
    def __init__(self):
        self.loss_fn = 'BCE_loss'
        self.measure = 'auroc'
        self.train_obj = 'TrainGAN'
        self.eval_obj = 'EvalGAN'
        self.netG = Generator()
        self.netD = Discriminator()