import torch
import torch.nn.functional as F
import torchaudio

#===================================================================================================
# Матожидание заменяется на среднее по батчу
#===================================================================================================

def discriminator_loss(original_batch, decoded_batch):
    # В статье SoundStream формула (1)
    # max между 0 и чем-то просто перепишем через relu
    loss = 0.0
    for original_feats, decoded_feats in zip(original_batch, decoded_batch):
        original_logits = original_feats[-1]
        decoded_logits = decoded_feats[-1]
        loss += F.relu(1 - original_logits).mean() + F.relu(1 + decoded_logits).mean()
    return loss / len(original_batch)

def adversarial_generator_loss(decoded_batch):
    # В статье SoundStream формула (2)
    loss = 0.0
    for decoded_feats in decoded_batch:
        decoded_logits = decoded_feats[-1]
        loss += F.relu(1 - decoded_logits).mean()
    return loss / len(decoded_batch)
    

def feature_matching_loss(original_batch, decoded_batch):
    # В статье SoundStream формула (3)
    loss = 0.0
    n = 0 # это будет KL 
    for original_feats, decoded_feats in zip(original_batch, decoded_batch):
        for original_feat, decoded_feat in zip(original_feats, decoded_feats):
            loss += torch.mean(torch.abs(original_feat.detach() - decoded_feat))
            n += 1
    return loss / n if n > 0 else 0.0


def multi_scale_spectral_reconstruction_loss(x, x_hat, sample_rate=16000, n_mels=64):
    # В статье SoundStream формула (4) и (5)
    # where S_t^s (x) denotes the t-th frame of a 64-bin melspectrogram computed with window length equal to s and hop length equal to s/4
    # We set \alpha_s = \sqrt{s/2} as in [48]

    scales = [64, 128, 256, 512, 1024, 2048]
    loss = 0.0

    x = x.squeeze(1)
    x_hat = x_hat.squeeze(1)

    for s in scales:
        hop = s // 4
        alpha = (s / 2) ** 0.5  # \alpha_s из статьи

        window = torch.hann_window(s).to(x.device)

        def mel_spec(wav, s=s, hop=hop, window=window):
            stft = torch.stft(wav, n_fft=s, hop_length=hop,
                              win_length=s, window=window,
                              return_complex=True)
            power = stft.abs() ** 2

            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=s // 2 + 1,
                f_min=0.0,
                f_max=float(sample_rate) / 2.0,
                n_mels=n_mels,
                sample_rate=sample_rate,
            ).to(x.device)

            mel = torch.matmul(mel_fb.T, power)
            return mel
 
        S_x = mel_spec(x)
        S_xhat = mel_spec(x_hat)
 
        loss += torch.mean(torch.abs(S_x - S_xhat))
        log_S_x = torch.log(S_x.clamp(min=1e-5))
        log_S_xhat = torch.log(S_xhat.clamp(min=1e-5))
        loss += alpha * torch.mean((log_S_x - log_S_xhat) ** 2)
 
    return loss
 
def generator_loss(x, x_hat, original_outputs, decoded_outputs, commitment_loss):
    # В статье SoundStream формула (6)
    l_adv = adversarial_generator_loss(decoded_outputs)
    l_feat = feature_matching_loss(original_outputs, decoded_outputs)
    l_rec = multi_scale_spectral_reconstruction_loss(x, x_hat)
    # Цитируем статью (сразу после формулы (6)):
    # In all our experiments we set \lambda_adv = 1, \lambda_feat = 100 and \lambda_rec = 1.
    total = 1.0 * l_adv + 100.0 * l_feat + 1.0 * l_rec + 1.0 * commitment_loss
    # Цитируем задание:
    # For training such a complicated model, it is useful to not only log the joint loss, but also the individual terms to monitor 
    # if something is going wrong. **You must provide these individual logs**.
    logs = {
        "loss_g_total": total.item(),
        "loss_g_adv": l_adv.item(),
        "loss_g_feat": l_feat.item(),
        "loss_g_rec": l_rec.item(),
        "loss_g_commit": commitment_loss.item(),
    }
    return total, logs

