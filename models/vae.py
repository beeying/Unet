import torch
import torch.nn as nn

ENCODER_IN = [1, 64, 64, 64, 64, 128, 128, 256, 128]
ENCODER_OUT = [64, 64, 64, 64, 128, 128, 256, 128, 64]
ENCODER_KERNEL = [4, 4, 4, 3, 4, 3, 4, 3, 3]
ENCODER_STRIDE = [2, 2, 2, 1, 2, 1, 2, 1, 1]
ENCODER_PADDING = [1, 1, 1, 1, 1, 1, 1, 1, 1]

DECODER_IN = [500, 64, 64, 256, 128, 128, 64, 64, 64, 64]
DECODER_OUT = [64, 64, 256, 128, 128, 64, 64, 64, 64, 1]
DECODER_KERNEL = [8, 3, 3, 3, 3, 3, 3, 3, 3, 3]
DECODER_STRIDE = [1, 1, 1, 2, 1, 2, 1, 2, 2, 2]
DECODER_PADDING = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
DECODER_OUT_PADDING = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]


class VAE(nn.Module):
    def __init__(self):
        self.loss_fn = 'vae_loss'
        super().__init__()
        self.encoder = self._make_conv_layers(ENCODER_IN, ENCODER_OUT, ENCODER_KERNEL, ENCODER_STRIDE, ENCODER_PADDING)
        self.decoder = self._make_deconv_layers(DECODER_IN, DECODER_OUT, DECODER_KERNEL, DECODER_STRIDE, DECODER_PADDING,
                                                DECODER_OUT_PADDING)
        self.linear_mu = nn.Conv2d(64, 500, kernel_size=8,
                          stride=1,padding=0, bias=True)
        self.linear_var = nn.Conv2d(64, 500, kernel_size=8,
                                   stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_conv_layers(self, inplanes, outplanes, kernel, stride, padding):
        modules = []
        for i in range(len(inplanes)):
            modules.extend([
                nn.Conv2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                          stride=stride[i],
                          padding=padding[i], bias=True),
                nn.LeakyReLU(0.2)])
        return nn.Sequential(*modules)

    def _make_deconv_layers(self, inplanes, outplanes, kernel, stride, padding, output_padding):
        modules = []
        for i in range(len(inplanes)):
            modules.extend([
                nn.ConvTranspose2d(inplanes[i], outplanes[i], kernel_size=kernel[i],
                                   stride=stride[i], padding=padding[i], output_padding=output_padding[i], bias=True),
                nn.LeakyReLU(0.2)])
        modules.extend([nn.Sigmoid()])
        return nn.Sequential(*modules)

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.encoder(x)
        mu = self.linear_mu(x)
        log_var = self.linear_var(x)
        z = self._reparameterize(mu, log_var)
        y = self.decoder(z)
        return {'pred': y , 'mu': mu, 'log_var': log_var}



