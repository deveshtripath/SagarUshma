"""
Reconstructed to match the real checkpoint (last_model.pt) exactly, based on
the state_dict key/shape dump from predictor/test.py. The *architecture*
below is verified correct - every layer's shape was traced against your
dump, so load_state_dict should succeed key-for-key. A few things are
still assumptions (I don't have your original training notebook) and
should be confirmed / swapped in:

  - PATCH_SIZE (16) / OUTPUT_SIZE (64): inferred from the two 4x4/stride-2
    ConvTranspose2d layers in the decoder (16 -> 32 -> 64), matching the
    "[64,64] output" from earlier. Change PATCH_SIZE if your real patches
    were a different size (must satisfy OUTPUT_SIZE = PATCH_SIZE * 4).
  - "current" and "wind" have 3 input channels each (not 2): assumed to be
    [u, v, speed] where speed = sqrt(u^2 + v^2).
  - "sst" has 3 input channels: currently just the same static SST value
    repeated 3x as a placeholder - swap in the real 3 channels (e.g.
    value/anomaly/trend) once you have that logic.
  - Decoder output has 2 channels, labelled OUTPUTS below as "temperature"
    and "salinity" (GLORYS-style targets) - confirm/rename against your
    actual training target.
  - This checkpoint has no x_mean/x_std/y_mean/y_std, so (as before) no
    normalization is applied.
"""
import torch
import torch.nn as nn

MODALITY_CHANNELS = {"sst": 3, "sss": 1, "ssh": 1, "current": 3, "wind": 3}
MODALITY_ORDER = ["sst", "sss", "ssh", "current", "wind"]
FEATURE_DIM = 32              # per-modality encoder output channels
FUSED_DIM = 128                # channels after fusion / temporal hidden size
PATCH_SIZE = 16                # spatial size fed into the encoders
OUTPUT_SIZE = PATCH_SIZE * 4   # decoder upsamples x2 twice -> 64
OUTPUTS = ["temperature", "salinity"]  # ASSUMPTION - confirm against training code


class ModalityEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class ModalityBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.sst_encoder = ModalityEncoder(MODALITY_CHANNELS["sst"])
        self.sss_encoder = ModalityEncoder(MODALITY_CHANNELS["sss"])
        self.ssh_encoder = ModalityEncoder(MODALITY_CHANNELS["ssh"])
        self.current_encoder = ModalityEncoder(MODALITY_CHANNELS["current"])
        self.wind_encoder = ModalityEncoder(MODALITY_CHANNELS["wind"])

    def forward(self, inputs):
        return [
            self.sst_encoder(inputs["sst"]),
            self.sss_encoder(inputs["sss"]),
            self.ssh_encoder(inputs["ssh"]),
            self.current_encoder(inputs["current"]),
            self.wind_encoder(inputs["wind"]),
        ]


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        n = len(MODALITY_ORDER)
        self.gate = nn.Sequential(
            nn.Conv2d(n * FEATURE_DIM, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n, 1),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(FEATURE_DIM, FUSED_DIM, 3, padding=1),
            nn.BatchNorm2d(FUSED_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        concat = torch.cat(feats, dim=1)           # (B, 5*32, H, W)
        gates = self.gate(concat)                  # (B, 5, H, W)
        fused = sum(gates[:, i:i + 1] * feats[i] for i in range(len(feats)))
        return self.output(fused)                  # (B, 128, H, W)


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch=FUSED_DIM, hidden_ch=FUSED_DIM):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, 3, padding=1)
        self.candidate = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)

    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        update, reset = self.gates(combined).chunk(2, dim=1)
        update, reset = torch.sigmoid(update), torch.sigmoid(reset)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1 - update) * hidden + update * candidate


class TemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = ConvGRUCell()

    def forward(self, sequence):
        b, _, h, w = sequence[0].shape
        hidden = torch.zeros(b, FUSED_DIM, h, w, device=sequence[0].device, dtype=sequence[0].dtype)
        for frame in sequence:
            hidden = self.cell(frame, hidden)
        return hidden


class DepthEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, depth):
        return self.network(depth)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_embedding = DepthEmbedding()
        self.depth_projection = nn.Linear(32, 32)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(FUSED_DIM, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(32, len(OUTPUTS), 1)

    def forward(self, hidden, depth):
        depth_mod = self.depth_projection(self.depth_embedding(depth))  # (B, 32)
        x = self.up1(hidden)
        x = self.up2(x)
        x = x + depth_mod[:, :, None, None]
        x = self.refine(x)
        return self.output(x)   # (B, len(OUTPUTS), 64, 64)

class OceanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modalities = ModalityBank()
        self.fusion = GatedFusion()
        self.temporal = TemporalEncoder()
        self.decoder = Decoder()

    def encode(self, sequence_inputs):
        """Run the modality encoders + fusion + temporal cell ONCE.
        The result (hidden state) can be reused for many depths, since
        only the decoder is depth-conditioned - avoids re-running the
        expensive part of the model per depth level."""
        fused = [self.fusion(self.modalities(day)) for day in sequence_inputs]
        return self.temporal(fused)

    def decode(self, hidden, depth):
        return self.decoder(hidden, depth)

    def forward(self, sequence_inputs, depth):
        """Kept for convenience/back-compat - single depth in one call."""
        return self.decode(self.encode(sequence_inputs), depth)