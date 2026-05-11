import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================================================
# Архитектура энкодера взята из статьи SoundStream. Ориентация на Fig. 3: Encoder and decoder model architecture.
# ====================================================================================================
# Цитируем статью:
# To guarantee real-time inference, all convolutions are causal.
# This means that padding is only applied to the past but not the future in both training and offline inference, 
# whereas no padding is used in streaming inference. 
# We use the ELU activation [42] and we do not apply any normalization.
# ====================================================================================================
# По статье все используемые в SoundStream свёртки - это обычные nn.Conv1d, но с каузальностью, то есть с paddingом только в прошлое
# ====================================================================================================
# Цитируем условие задания:
# 6.We use sample rate 16kHz, so we need to adjust the strides to preserve the bitrate around 6k. 
#   Use [2, 4, 5, 5]. - подразумеваются значения stides для свёрток в энкодере, и, в обратном порядке, для декодера.
# 2.We do not care about denoising here, so do not add the FiLM module used in the original paper.
# Мы последуем этим пунктам, и не используем FiLM, а также используем указанные strides в энкодере и декодере.
# =====================================================================================================
# Из статьи так же знаем, что на выходе энекодера будет матрица, в py_torch нотации, S х D, где S - это длина вектора выхода энкодера,
# которую мы можем вычислить как S =  CROP_SAMPLES // (2 * 4 * 5 * 5) = 8000 // 200 = 40, а D - это число каналов на выходе энкодера.
# На схеме Fig. 3 это число называется K, мы будем использовать D аналогично rvq.py, чтобы было одинаково. 
# =====================================================================================================




# ====================================================================================================
# Мы реализуем обертку над nn.Conv1d, чтобы получить казуальную версию для энкодера
# Входных каналов в первой свёртке энкодера - 1, так как аудио моно
# ====================================================================================================
# Важное уточнение (для меня важное, потому что я запуталась)
# =====================================================================================================
# Чтобы потом смочь построить обратную свертку, нужно понять как меняются размерности
# Помним, что output_size = ((input_size + (padding_past + padding_future) - dilation*(kernel_size-1) - 1) // stride) + 1
# Тогда в forward (padding_past, padding_future) = ((kernel_size - 1) * dilation, 0) и получаем, что
# output_size = (input_size - 1) // stride + 1
# Транспонированная Conv1d из декодера должен превращать output_size в input_size
# Это нужно, чтобы согласовать размерности
# =====================================================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            dilation=dilation,
            padding = 0
        )
        
    def forward(self, x):
        x = F.pad(x, (self.padding, 0)) # паддинг только в прошлое для каузальности
        x = self.conv(x)
        return x


# =====================================================================================================
#  --------------------------
# | ResidualUnit (N, dilation)| ---
#  --------------------------      | и Residual Connection
#   Conv1D (k=7, n=N, dilation)    | простите за лишний мусор
#                                  | это параноидальное
#   Conv1D (k=1, n=N)              |
#               <------------------
# =====================================================================================================

class ResidualUnit(nn.Module):
    def __init__(self, N, dilation):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels=N, out_channels=N, kernel_size=7, dilation=dilation),
            nn.ELU(),
            CausalConv1d(in_channels=N, out_channels=N, kernel_size=1)
        )

    def forward(self, x):
        return x + self.net(x)
    

# =====================================================================================================
# -------------------------
#    EncoderBlock (N, S)   |
# -------------------------
# ResidualUnit (N/2, dilation=1)
# ResidualUnit (N/2, dilation=3)
# ResidualUnit (N/2, dilation=9)
# Conv1D (k=2S, n=N, stride=S)
# =====================================================================================================

class EncoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(N // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(N // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(N // 2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=N//2, out_channels=N, kernel_size=2 * S, stride=S)
        )

    def forward(self, x):
        return self.net(x)
    

# =====================================================================================================
# ------------------------
#         Encoder         |
# ------------------------
#   Waveform @ 24 kHz
# Conv1D (k=7, n=C)
# EncoderBlock (N=2C, S=2)
# EncoderBlock (N=4C, S=4)
# EncoderBlock (N=8C, S=5)
# EncoderBlock (N=16C, S=8)
# Conv1D (k=3, n=K)
# FİLM conditioning
# Embeddings @ 75 Hz
# =====================================================================================================

class Encoder(nn.Module):
    def  __init__(self, C, D):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(N = 2 * C, S = 2),
            nn.ELU(),
            EncoderBlock(N = 4 * C, S = 4),
            nn.ELU(),
            EncoderBlock(N = 8 * C, S = 5),
            nn.ELU(),
            EncoderBlock(N = 16 * C, S = 5),
            nn.ELU(),
            CausalConv1d(in_channels=16 * C, out_channels=D, kernel_size=3),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================================================
# Вспоминаем уточнение про размерности декодера
# Тогда для транспонированной меняем местами размеры входа и выхода
# Получаем input_size - 1 = (out_size - 1) // stride
# Нужно для подбора обрезки
# =====================================================================================================

class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.trim = kernel_size - stride  #kernel=2S, stride=S, а значит trim=S
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )

    def forward(self, x):
        x = self.conv(x)
        if self.trim > 0:
            x = x[:, :, :-self.trim]
        return x
    

# =====================================================================================================
# -------------------------
#    DecoderBlock (N, S)   |
# -------------------------
# (Conv1D)' (k=2S, n=N, stride=S)
# ResidualUnit (N/2, dilation=1)
# ResidualUnit (N/2, dilation=3)
# ResidualUnit (N/2, dilation=9)
# =====================================================================================================

class DecoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.net = nn.Sequential(
            CausalConvTranspose1d(in_channels=N, out_channels=N // 2, kernel_size=2 * S, stride=S),
            nn.ELU(),
            ResidualUnit(N // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(N // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(N // 2, dilation=9)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================================================
# Декодер - это просто энкодер в обратном порядке, с транспонированными свёртками и обратными strides
# FiLM в декодере мы не используем, так как по условию задания не используем его вообще
# =====================================================================================================
# На картинке Fig. 3 в декодере для меня происходит небольшая чушь в размерностях, потому что они не сходятся,
# или сходятся тогда, когда формальность нужно опустить и верить, что имелось в виду количество входов, а не выходов
# Ниже схема, если не учитывать корректировку Hz и FiLM.
# =====================================================================================================
# ------------------------
#         Decoder         |
# ------------------------
#   Embeddings @ 75 Hz
#   FiLM conditioning
#   Conv1D (k=7, n=16C)
#   DecoderBlock (N=8C, S=8)
#   DecoderBlock (N=4C, S=5)
#   DecoderBlock (N=2C, S=4)
#   DecoderBlock (N=C, S=2)
#   Conv1D (k=7, n=1)
#   Waveform @ 24 kHz
# =====================================================================================================

class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16 * C, kernel_size=7),
            nn.ELU(),
            DecoderBlock(N = 16 * C, S = 5), # не 8, а 16
            nn.ELU(),
            DecoderBlock(N = 8 * C, S = 5), # не 4, а 8
            nn.ELU(),
            DecoderBlock(N = 4 * C, S = 4), # не 2, а 4
            nn.ELU(),
            DecoderBlock(N = 2 * C, S = 2), # не 1, а 2
            nn.ELU(),   
            CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )
    
    def forward(self, x):
        return self.net(x)
    