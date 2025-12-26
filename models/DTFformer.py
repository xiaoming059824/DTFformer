import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
import math
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



class FreqBranch(nn.Module):
    def __init__(self, configs, patch_len):
        super(FreqBranch, self).__init__()
        self.d_model = configs.d_model
        self.patch_len = patch_len
        self.num_freqs = (patch_len // 2) + 1

        self.freq_importance = nn.Sequential(
            nn.Linear(self.num_freqs, self.num_freqs),
            nn.Sigmoid()
        )

        self.adaptive_filter = nn.Parameter(
            torch.ones(1, 1, self.num_freqs)
        )

        self.real_proj = nn.Linear(self.num_freqs, self.d_model // 2)
        self.imag_proj = nn.Linear(self.num_freqs, self.d_model // 2)

        self.encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU() if configs.activation == 'gelu' else nn.ReLU(),
            nn.LayerNorm(self.d_model),
            nn.Dropout(configs.dropout)
        )


    def forward(self, patches):
        # patches shape: [B*C, NumPatches, patch_len]
        fft_vals = torch.fft.rfft(patches, dim=-1)  # [B*C, NumPatches, num_freqs]

        fft_amp = torch.abs(fft_vals)
        # importance shape: [B*C, NumPatches, num_freqs]
        importance = self.freq_importance(fft_amp)

        fft_real_filtered = fft_vals.real * importance * self.adaptive_filter
        fft_imag_filtered = fft_vals.imag * importance * self.adaptive_filter

        real_feat = self.real_proj(fft_real_filtered)
        imag_feat = self.imag_proj(fft_imag_filtered)

        freq_feat_proj = torch.cat([real_feat, imag_feat], dim=-1)

        freq_feat = self.encoder(freq_feat_proj)

        return freq_feat


class FreqEnhancedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FreqEnhancedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)

        self.q_bias_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.k_gate_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.v_bias_projection = nn.Linear(d_model, self.d_values * n_heads)

        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, attn_mask, context=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        Q = self.query_projection(queries).view(B, L, H, self.d_keys)
        K = self.key_projection(keys).view(B, S, H, self.d_keys)
        V = self.value_projection(values).view(B, S, H, self.d_values)

        if context is not None:
            q_bias = self.q_bias_projection(context).view(B, L, H, self.d_keys)
            Q = Q + q_bias
            k_gate = torch.sigmoid(self.k_gate_projection(context).view(B, S, H, self.d_keys))
            K = K * k_gate
            v_bias = self.v_bias_projection(context).view(B, S, H, self.d_values)
            V = V + v_bias

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_keys ** 0.5)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(attn_weights), V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_projection(out), attn_weights


class FreqEnhancedAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(FreqEnhancedAttentionLayer, self).__init__()
        self.inner_attention = attention
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attn_mask, context=None):
        out, attn = self.inner_attention(queries, keys, values, attn_mask, context=context)
        return self.norm(queries + out), attn


class FreqEnhancedEncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(FreqEnhancedEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, context=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, context=context)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class GatedInteractionLayer(nn.Module):
    def __init__(self, d_model):
        super(GatedInteractionLayer, self).__init__()
        self.gate_t = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.gate_g = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_g = nn.LayerNorm(d_model)

    def forward(self, time_feat, guided_feat):
        combined = torch.cat([time_feat, guided_feat], dim=-1)

        gate_for_time = self.gate_t(combined)
        gate_for_guided = self.gate_g(combined)

        next_time_feat = self.norm_t(time_feat + gate_for_time * guided_feat)
        next_guided_feat = self.norm_g(guided_feat + gate_for_guided * time_feat)

        return next_time_feat, next_guided_feat


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.patch_len = getattr(configs, 'patch_len', 48)
        self.stride = getattr(configs, 'stride', 24)

        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        padding = self.stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        self.freq_branch = FreqBranch(configs, self.patch_len)

        self.time_to_fre_encoder_layers = nn.ModuleList([
            FreqEnhancedEncoderLayer(
                FreqEnhancedAttentionLayer(
                    FreqEnhancedAttention(configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(self.e_layers)
        ])

        self.guided_encoder_layers = nn.ModuleList([
            FreqEnhancedEncoderLayer(
                FreqEnhancedAttentionLayer(
                    FreqEnhancedAttention(configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(self.e_layers)
        ])

        self.interaction_layers = nn.ModuleList([
            GatedInteractionLayer(configs.d_model) for _ in range(self.e_layers)
        ])

        self.dropout = nn.Dropout(configs.dropout)

        self.norm_time = nn.LayerNorm(configs.d_model)
        self.norm_guided = nn.LayerNorm(configs.d_model)

        num_patches = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = (configs.d_model * 2) * num_patches

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev

        x_permuted = x_enc_norm.permute(0, 2, 1)
        temporal_embeddings, n_vars = self.patch_embedding(x_permuted)

        padding_layer = nn.ReplicationPad1d((0, self.stride))
        x_padded = padding_layer(x_permuted)
        x_unfolded = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = x_unfolded.shape[2]
        patches_for_freq = x_unfolded.reshape(-1, num_patches, self.patch_len)
        freq_context = self.freq_branch(patches_for_freq)

        time_features = temporal_embeddings
        guided_features = freq_context

        for i in range(self.e_layers):
            time_features_out, _ = self.guided_encoder_layers[i](time_features, context=guided_features)
            guided_features_out, _ = self.time_to_fre_encoder_layers[i](guided_features, context=time_features)

            time_features, guided_features, _ = self.interaction_layers[i](time_features_out, guided_features_out)

        time_features = self.norm_time(time_features)
        guided_features = self.norm_guided(guided_features)

        fused_features = torch.cat([time_features, guided_features], dim=-1)

        fused_features = fused_features.reshape(-1, n_vars, num_patches, self.d_model * 2)
        fused_features = fused_features.permute(0, 1, 3, 2)
        dec_out = self.head(fused_features)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

        print(f"Task {self.task_name} not implemented in this simplified model.")
        return None