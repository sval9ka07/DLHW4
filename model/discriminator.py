import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================================================
# Цитируем статью SoundStream:
# To compute the adversarial losses described in Section III-E, we define two different discriminators:
# i) a wave-based discriminator, which receives as input a single waveform; 
# ii) an STFT-based discriminator, which receives as input the complexvalued STFT of the input waveform, expressed in terms of real and imaginary parts.
# Since both discriminators are fully convolutional, the number of logits in the output is proportional to the length of the input audio.
# То есть мы научимся обрабатывать аудио любой длины и видимо должны сами с помощью padding регулировать длину аудио, то есть padding = kernel_size // 2.
# ====================================================================================================
# В статье указано, что WaveDiscriminator - это архитектура из статьи MelGAN - [15]:
# For the wave-based discriminator, we use the same multiresolution convolutional discriminator proposed in [15] and adopted in [45].
# В статье MelGAN написано,что:
# Multi-Scale Architecture Following Wang et al. (2018b), 
# we adopt a multi-scale architecture with 3 discriminators (D1,D2,D3) 
# that have identical network structure but operate on different audio scales.
# The downsampling is performed using strided average pooling with kernel size 4. 
# То есть, что мы будем использовать AveragePool1d(kernel_size=4, stride=2, padding=1) между блоками дискриминатора, чтобы получать разные масштабы аудио.
# ====================================================================================================
# В аппендиксе статьи MelGAN указано, что архитектура одного блока дискриминатора
# 15 x 1, stride=1 conv 16 IReLU
# 41 x 1, stride=4 groups=4 conv 64 IReLU
# 41 x 1, stride=4 groups=16 conv 256 IReLU
# 41 x 1, stride=4 groups=64 conv 1024 IReLU
# 41 x 1, stride=4 groups=256 conv 1024 IReLU
# 5 x 1, stride=1 conv 1024 IReLU
# 3 x 1, stride=1 conv 1
# (b) Discriminator Block Architecture
# Однако, цитируя условие задания:
# Discriminator uses LeakyReLU with 0.2 slope, то есть мы должны заменить IReLU на LeakyReLU(0.2).
# ====================================================================================================

class DiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels= 16, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 16, out_channels= 64, kernel_size=41, stride=4, groups=4, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 64, out_channels= 256, kernel_size=41, stride=4, groups=16, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 256, out_channels= 1024, kernel_size=41, stride=4, groups=64, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 1024, out_channels= 1024, kernel_size=41, stride=4, groups=256, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 1024, out_channels= 1024, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 1024, out_channels= 1, kernel_size=3, stride=1, padding=1)  
        )
    
    def forward(self, x):
        # upd: нужны фичи для feature matching loss, сохраняем не только последнее
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        return features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorBlock(),
            DiscriminatorBlock(),
            DiscriminatorBlock()
        ])
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        results = []
        results.append(self.discriminators[0](x))
        x = self.downsample(x)
        results.append(self.discriminators[1](x))
        x = self.downsample(x)
        results.append(self.discriminators[2](x))
        return results

# =====================================================================================================
# Смотрим на схему из статьи SoundStream Fig. 4: STFT-based discriminator architecture.
#  ----------------------
# | STFTDiscriminator(C) |
#  ----------------------
# Waveform @ 24 kHz
# STFT (w, h)
# ResidualUnit (N=C, m=2, s=(1, 2))
# ResidualUnit (N=2C, m=2, s=(2, 2))
# ResidualUnit (N=4C, m=1, s=(1, 2))
# ResidualUnit (N=4C, m=2, s= (2, 2))
# ResidualUnit (N=8C, m=1, s=(1, 2))
# ResidualUnit (N=8C, m=2, s=(2, 2))
# Conv2D (k=1xF/26 ,n=1)
# Logits
#  -------------------------------
# | ResidualUnit (N, m, (s_t, s_f)) |
#  -------------------------------
# Conv2D (k=3x3, n=N)
# Conv2D (k=(s_t+2)x(s_f+2), n=mN)
# ===================================================================================================== 
# КОСТЫЛЬ с shortcut, пока оставляем, кажется, что это распространенная практика
# =====================================================================================================

class ResidualUnit2D(nn.Module):
    def __init__(self, N, m, s):
        super().__init__()
        s_t, s_f = s
        self.net = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(N, m*N, kernel_size=(s_t+2, s_f+2), stride=s, 
                      padding=(s_t//2, s_f//2)) 
        )
        self.shortcut = nn.Conv2d(N, m*N, kernel_size=1, stride=s)

    def forward(self, x):
        short = self.shortcut(x)
        main = self.net(x)
        # подрезаем до минимального размера по каждой оси
        t = min(short.shape[2], main.shape[2])
        f = min(short.shape[3], main.shape[3])
        return short[:, :, :t, :f] + main[:, :, :t, :f]
    
# =====================================================================================================
# Снова цитируем статью
# The STFT-based discriminator is illustrated in Figure 4 and operates on a single scale, 
# computing the STFT with a window length of W = 1024 samples and a hop length of H = 256 samples.
# -----------------------------------------------------------------------------------------------------
# A 2D-convolution (with kernel size 7 × 7 and 32 channels) is followed by a sequence of residual blocks. 
# Each block starts with a 3×3 convolution, followed by a 3×4 or a 4×4 convolution, with strides equal to (1, 2) or (2, 2), 
# where (s_t, s_f ) indicates the down-sampling factor along the time axis and the frequency axis. We alternate between (1, 2) and (2, 2) strides, for a total of 6 residual blocks. 
# То есть нужно еще одну свертку добавить в начале.
# The number of channels is progressively increased with the depth of the network. 
# At the output of the last residual block, the activations have shape T /(H · 23) × F/26, 
# where T is the number of samples in the time domain and F = W/2 is the number of frequency bins. 
# The last layer aggregates the logits across the (down-sampled) frequency bins with a fully connected layer (implemented as a 1 × F/26 convolution), 
# to obtain a 1-dimensional signal in the (down-sampled) time domain.
# =====================================================================================================
class STFTDiscriminator(nn.Module):
    # Этот дискриминатор работает со спектограммой, чтобы видеть гармоническую структуру
    # А-ля чтобы чувствовать "музыкальность", так как подобное отличает голос человека от голоса робота
    def __init__(self, C, W = 1024, H = 256):
        super().__init__()
        self.W = W
        self.H = H
        F = W // 2

        self.net = nn.Sequential(
            nn.Conv2d(2, C, kernel_size=(7, 7), padding=(3, 3)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(C, 2, (1, 2)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(2 * C, 2, (2, 2)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(4 * C, 1, (1, 2)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(4 * C, 2, (2, 2)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(8 * C, 1, (1, 2)),
            nn.LeakyReLU(0.2),
            ResidualUnit2D(8 * C, 2, (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels= 2 * 8 * C, out_channels=1, kernel_size= (1, F // 64))
        )

    def forward(self, x):
        # убираем лишнюю размерность, чтобы работать с матрицей (у нас моно аудио)
        x = x.squeeze(1)
        # преобразование фурье
        stft = torch.stft(x, n_fft=self.W, hop_length=self.H, 
                        return_complex=True, window=torch.hann_window(self.W).to(x.device))
        # раскладываем на вещественную и мнимую части
        stft = torch.stack([stft.real, stft.imag], dim=1)
        stft = stft.transpose(2, 3)
        features = []
        for layer in self.net:
            stft = layer(stft)
            features.append(stft)
        return features

