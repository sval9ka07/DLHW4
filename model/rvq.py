import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================================================
# Обозначения взяты из статьи SoundStream:
# N_q — число кодбуков в RVQ
# N — количество паттернов в каждом codebook
# D — размерность паттерна, а так же количество каналов в выходе энкодера
# S - длина вектора выхода энкодера (длина аудио в семплах, делённая на фактор downsampling в энкодере)
# ====================================================================================================
# Цитируем статью:
# To address this issue we adopt a Residual Vector Quantizer , which cascades N_q layers of VQ as follows.
# The unquantized input vector is passed through a first VQ and quantization residuals are computed. The residuals are then
# iteratively quantized by a sequence of additional N_q − 1 vector quantizers, as described in Algorithm 1.
# ----------------------------------------------------------------------------------------------------
# Algorithm 1: Residual Vector Quantization
# ----------------------------------------------------------------------------------------------------
# Input: y = enc(x) the output of the encoder, vector
# quantizers Qi for i = 1..Na
# Output: the quantized
# y_hat <- 0.0
# residual <- y
# for i = 1 to N_q do
#   y_hat += Q_i(residual)
#   residual -= Q_i(residual)
# return y_hat
# ====================================================================================================
# Цитируем условие задания:
# 5. Add commitment loss (MSE between encoder output and its quantized detached version) to avoid encoder outputs drifting from the codebook. 
# Use 1.0 weight for the loss.
# ====================================================================================================
# Цитируем статью:
# The codebook of each quantizer is trained with exponential
# moving average updates, following the method proposed in
# VQ-VAE-2 [32]. 
# Мы не будем это реализовывать.
# Это из статьи - Enabling bitrate scalability with quantizer dropout - мы тоже не будем реализовывать.
# Но тут нас поддерживает задание:
# 3. We only focus on a single bitrate, so do not add bitrate dropout.
# ====================================================================================================
class VectorQuantizer(nn.Module):
    def __init__(self, N, D):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(N, D))

    def quantize(self, encoder_output_batch):
        B, D, S = encoder_output_batch.shape 
        encoder_output_batch = encoder_output_batch.permute(0, 2, 1)
        # Тензор расстояний между каждым вектором из encoder_output_batch и каждым паттерном из codebook
        distances = torch.cdist(encoder_output_batch, self.codebook)
        nearest_indices = torch.argmin(distances, dim=-1)
        quantized = self.codebook[nearest_indices]
        # Важно, что мы избегаем проблем с градиентами, используя STE
        # Мы считаем loss, будто бы дискретное приближение с помощью codebook это и есть настоящий выход энкодера
        quantized_STE = encoder_output_batch + (quantized - encoder_output_batch).detach()
        quantized_STE = quantized_STE.permute(0, 2, 1)
        # По условию задания добавляем commitment loss с весом 1.0
        # Этот loss позволяет штрафовать энкодер за то, что он выдаёт вектора, которые плохо аппроксимируются codebook
        commitment_loss = F.mse_loss(quantized.detach(), encoder_output_batch)
        loss = F.mse_loss(quantized, encoder_output_batch.detach()) + commitment_loss
        return nearest_indices, quantized_STE, loss
    
# В ResidualVectorQuantizer мы просто поочередно применяем VectorQuantizerы, каждый из которых кодирует остаток от предыдущего
class ResidualVectorQuantizer(nn.Module):
    def __init__(self, N_q, N, D):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer(N, D) for _ in range(N_q)])

    def forward(self, encoder_output):
        residual = encoder_output
        all_indices = list()
        full_loss = 0
        for quantizer in self.quantizers:
            nearest_indices, quantized_STE, loss = quantizer.quantize(residual)
            all_indices.append(nearest_indices)
            residual = residual - quantized_STE
            full_loss += loss
        # Из всего вычли шум, который не смогли заквантовать, вместо того, чтобы просто сложить все quantized_STE
        # С точки зрения формулы это одно и то же
        y_hat = encoder_output - residual
        return all_indices, y_hat, full_loss