import math

import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels, kernel, stride=(1, 1), padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding), nn.LeakyReLU(0.01))


def initialize_keras_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.LSTMCell)):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
                if "bias_ih" in name:
                    with torch.no_grad():
                        parameter[module.hidden_size:2 * module.hidden_size].fill_(1)
                else:
                    parameter.requires_grad_(False)  # Keras has a single LSTM bias


class BilinearNormalization(nn.Module):
    def __init__(self, features, time_steps, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.feature_scale = nn.Parameter(torch.empty(features, 1))
        self.feature_bias = nn.Parameter(torch.zeros(features, 1))
        self.time_scale = nn.Parameter(torch.empty(time_steps, 1))
        self.time_bias = nn.Parameter(torch.zeros(time_steps, 1))
        self.feature_log_weight = nn.Parameter(torch.tensor(math.log(math.expm1(0.5))))
        self.time_log_weight = nn.Parameter(torch.tensor(math.log(math.expm1(0.5))))
        nn.init.xavier_normal_(self.feature_scale)
        nn.init.xavier_normal_(self.time_scale)

    def forward(self, x):
        temporal_mean = x.mean(dim=2, keepdim=True)
        temporal_std = x.std(dim=2, keepdim=True)
        temporal_std = torch.where(temporal_std < self.epsilon, torch.ones_like(temporal_std), temporal_std)
        temporal = self.feature_scale * ((x - temporal_mean) / temporal_std) + self.feature_bias
        feature_mean = x.mean(dim=1, keepdim=True)
        feature_std = x.std(dim=1, keepdim=True)
        feature_std = torch.where(feature_std < self.epsilon, torch.ones_like(feature_std), feature_std)
        feature = self.time_scale.T * ((x - feature_mean) / feature_std) + self.time_bias.T
        feature_weight = F.softplus(self.feature_log_weight)
        time_weight = F.softplus(self.time_log_weight)
        return feature_weight * feature + time_weight * temporal


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = x
        x = self.net(x)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        return F.gelu(self.norm(x))


class DeepLOB(nn.Module):
    def __init__(self, channels=32, inception_channels=64, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(conv(1, channels, (1, 2), (1, 2)), conv(channels, channels, (4, 1), padding="same"), conv(channels, channels, (4, 1), padding="same"))
        self.conv2 = nn.Sequential(conv(channels, channels, (1, 2), (1, 2)), conv(channels, channels, (4, 1), padding="same"), conv(channels, channels, (4, 1), padding="same"))
        self.conv3 = nn.Sequential(conv(channels, channels, (1, 10)), conv(channels, channels, (4, 1), padding="same"), conv(channels, channels, (4, 1), padding="same"))
        self.inception = nn.ModuleList([
            nn.Sequential(conv(channels, inception_channels, (1, 1), padding="same"), conv(inception_channels, inception_channels, (3, 1), padding="same")),
            nn.Sequential(conv(channels, inception_channels, (1, 1), padding="same"), conv(inception_channels, inception_channels, (5, 1), padding="same")),
            nn.Sequential(nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)), conv(channels, inception_channels, (1, 1), padding="same")),
        ])
        self.temporal_dropout = nn.Dropout1d(dropout)
        self.lstm = nn.LSTM(3 * inception_channels, 64, batch_first=True)
        self.classifier = nn.Linear(64, 3)
        self.apply(initialize_keras_weights)

    def forward(self, x):
        sequence = self.conv3(self.conv2(self.conv1(x.unsqueeze(1))))
        sequence = torch.cat([branch(sequence) for branch in self.inception], dim=1).squeeze(-1).transpose(1, 2)
        sequence = self.temporal_dropout(sequence.transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.lstm(sequence)[0][:, -1])


class TransformerLayer(nn.Module):
    def __init__(self, input_dim, heads, output_dim):
        super().__init__()
        expanded = input_dim * heads
        self.query = nn.Linear(input_dim, expanded)
        self.key = nn.Linear(input_dim, expanded)
        self.value = nn.Linear(input_dim, expanded)
        self.attention = nn.MultiheadAttention(expanded, heads, batch_first=True)
        self.project = nn.Linear(expanded, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = MLPBlock(input_dim, input_dim * 4, output_dim)

    def forward(self, x):
        residual = x
        x = self.attention(self.query(x), self.key(x), self.value(x), need_weights=False)[0]
        x = self.mlp(self.norm(self.project(x) + residual))
        return x + residual if x.shape[-1] == residual.shape[-1] else x


class TLOB(nn.Module):
    def __init__(self, lookback, hidden_dim=40, num_layers=4, num_heads=1, sinusoidal=True):
        super().__init__()
        self.bin = BilinearNormalization(40, lookback)
        self.embedding = nn.Linear(40, hidden_dim)
        steps = torch.arange(lookback, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / hidden_dim))
        position = torch.zeros(lookback, hidden_dim)
        position[:, 0::2] = torch.sin(steps * scale)
        position[:, 1::2] = torch.cos(steps * scale)
        position = position[None]
        if sinusoidal:
            self.register_buffer("position", position)
        else:
            self.position = nn.Parameter(torch.randn_like(position))
        self.layers = nn.ModuleList()
        feature_dim, time_dim = hidden_dim, lookback
        for index in range(num_layers):
            last = index == num_layers - 1
            next_feature = feature_dim // 4 if last else feature_dim
            next_time = time_dim // 4 if last else time_dim
            self.layers.append(TransformerLayer(feature_dim, num_heads, next_feature))
            self.layers.append(TransformerLayer(time_dim, num_heads, next_time))
            feature_dim, time_dim = next_feature, next_time
        flattened = feature_dim * time_dim
        final = []
        while flattened > 128:
            final += [nn.Linear(flattened, flattened // 4), nn.GELU()]
            flattened //= 4
        final.append(nn.Linear(flattened, 3))
        self.classifier = nn.Sequential(*final)

    def forward(self, x):
        x = self.bin(x.transpose(1, 2)).transpose(1, 2)
        x = self.embedding(x) + self.position
        for layer in self.layers:
            x = layer(x).transpose(1, 2)
        return self.classifier(x.transpose(1, 2).flatten(1))


class SpatialBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, width), nn.GELU(),
            nn.ZeroPad2d((1, 1, 2 * dilation, 0)),  # causal in time, symmetric over levels
            nn.Conv2d(width, width, 3, dilation=(dilation, 1), groups=width),
            nn.Conv2d(width, width, 1), nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, width), nn.GELU(),
            nn.ConstantPad1d((2 * dilation, 0), 0.0),
            nn.Conv1d(width, width, 3, dilation=dilation, groups=width),
            nn.Conv1d(width, width * 2, 1), nn.GLU(dim=1), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class SideEncoder(nn.Module):
    def __init__(self, spatial_width, temporal_width, dropout):
        super().__init__()
        self.input = nn.Linear(5, spatial_width)
        self.level_position = nn.Parameter(torch.randn(1, 1, 10, spatial_width) * 0.01)
        self.blocks = nn.Sequential(SpatialBlock(spatial_width, 1, dropout), SpatialBlock(spatial_width, 2, dropout))
        self.output = nn.Linear(10 * spatial_width, temporal_width)

    def forward(self, side):
        x = self.input(side) + self.level_position
        x = self.blocks(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.output(x.flatten(2))


class SymmetricLOB(nn.Module):
    def __init__(self, spatial_width=24, temporal_width=96, dropout=0.1, horizons=4):
        super().__init__()
        # how each column after the two sides moves under the bid/ask mirror: mid change, spread, then the "context" EWMAs (6 signed, 6 unsigned)
        self.register_buffer("global_signs", torch.tensor((-1.0, 1.0) + (-1.0,) * 6 + (1.0,) * 6), persistent=False)
        self.side_encoder = SideEncoder(spatial_width, temporal_width, dropout)
        self.fusion = nn.Sequential(nn.Linear(temporal_width * 4 + len(self.global_signs), temporal_width), nn.GELU(), nn.LayerNorm(temporal_width))
        self.temporal = nn.Sequential(*(TemporalBlock(temporal_width, dilation, dropout) for dilation in (1, 2, 4, 8, 16, 32)))
        self.horizon_query = nn.Parameter(torch.randn(horizons, temporal_width) * 0.02)
        self.horizon_decoder = nn.GRU(temporal_width * 2, temporal_width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(temporal_width * 2, temporal_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(temporal_width, 4))

    @staticmethod
    def _side(x, price, volume, flow):
        state = torch.stack((x[..., price], x[..., volume]), dim=-1)
        delta = torch.diff(state, dim=1, prepend=state[:, :1])
        return torch.cat((state, x[..., flow].unsqueeze(-1), delta), dim=-1)

    def _oriented(self, x):
        ask = self.side_encoder(self._side(x, slice(0, 10), slice(10, 20), slice(20, 30)))
        bid = self.side_encoder(self._side(x, slice(30, 40), slice(40, 50), slice(50, 60)))
        fused = self.fusion(torch.cat((ask, bid, bid - ask, bid * ask, x[..., 60:]), dim=-1))
        sequence = self.temporal(fused.transpose(1, 2)).transpose(1, 2)
        scores = torch.einsum("btd,hd->bht", sequence, self.horizon_query) / math.sqrt(sequence.shape[-1])
        context = torch.einsum("bht,btd->bhd", scores.softmax(-1), sequence)
        last = sequence[:, -1:, :].expand(-1, len(self.horizon_query), -1)
        query = self.horizon_query.unsqueeze(0).expand(len(x), -1, -1)
        decoded, _ = self.horizon_decoder(torch.cat((last, context), dim=-1))
        output = self.head(torch.cat((decoded, query), dim=-1))
        return output[..., :3], output[..., 3]

    def forward(self, x):
        logits, returns = self._oriented(x)
        mirrored = torch.cat((x[..., 30:60], x[..., :30], x[..., 60:] * self.global_signs), dim=-1)  # swap bid and ask, flip the signed columns
        reflected_logits, reflected_returns = self._oriented(mirrored)
        reflected_logits = reflected_logits[..., [2, 1, 0]]
        return (logits + reflected_logits) / 2, (returns - reflected_returns) / 2


class MultiHorizonDeepLOB(nn.Module):
    def __init__(self, hidden_dim=64, horizons=4):
        super().__init__()
        self.horizons = horizons
        backbone = DeepLOB(32, 64)
        self.conv1, self.conv2, self.conv3, self.inception = backbone.conv1, backbone.conv2, backbone.conv3, backbone.inception
        self.encoder = nn.LSTM(192, hidden_dim, batch_first=True)
        self.decoder = nn.LSTMCell(3 + hidden_dim, hidden_dim)
        self.context_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim, momentum=0.4, eps=1e-3) for _ in range(horizons))
        self.classifier = nn.Linear(2 * hidden_dim, 3)
        initialize_keras_weights(self.encoder)
        initialize_keras_weights(self.decoder)
        initialize_keras_weights(self.classifier)

    def forward(self, x):
        features = self.conv3(self.conv2(self.conv1(x.unsqueeze(1))))
        features = torch.cat([branch(features) for branch in self.inception], dim=1).squeeze(-1).transpose(1, 2)
        encoded, (hidden, cell) = self.encoder(features)
        hidden, cell = hidden[-1], cell[-1]
        previous = torch.zeros(len(x), 3, device=x.device, dtype=x.dtype)
        previous[:, 0] = 1
        context = hidden
        outputs = []
        for index in range(self.horizons):
            hidden, cell = self.decoder(torch.cat((previous, context), dim=1), (hidden, cell))
            attention = torch.bmm(encoded, hidden.unsqueeze(2)).squeeze(2).softmax(1)
            context = torch.bmm(attention.unsqueeze(1), encoded).squeeze(1)
            context = self.context_norms[index](context)
            logits = self.classifier(torch.cat((hidden, context), dim=1))
            outputs.append(logits)
            previous = logits.softmax(1)
        return torch.stack(outputs, dim=1)


class SiameseLOB(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        self.encoder = nn.LSTM(10, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim * num_layers, 64), nn.Linear(64, 1))

    def _encode(self, side):
        return self.encoder(side)[1][0].transpose(0, 1).flatten(1)

    def forward(self, x):
        difference = self._encode(x[..., 10:]) - self._encode(x[..., :10])
        return 2 * torch.sigmoid(self.head(difference)) - 1


class DeepOFILSTM(nn.Module):
    def __init__(self, horizons=4):
        super().__init__()
        self.lstm = nn.LSTM(20, 128, batch_first=True)
        self.head = nn.Linear(128, horizons)
        self.apply(initialize_keras_weights)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])


DEFAULTS = {"eps": 1e-8, "weight_decay": 0.0, "ema_decay": None, "class_weighted": False, "auxiliary_weight": 1.0,
            "min_delta": 0.0, "gradient_clip": None, "plateau_patience": None}

CONFIGS = {
    "deeplob": {
        "build": lambda horizons: DeepLOB(), "task": "classification", "multi": False, "features": "book", "lookback": 100,
        "optimizer": torch.optim.Adam, "lr": 1e-2, "eps": 1.0, "batch_size": 32, "max_epochs": 100,
        "monitor": "accuracy", "patience": 20,
    },
    "multihorizon_deeplob": {
        "build": lambda horizons: MultiHorizonDeepLOB(horizons=horizons), "task": "classification", "multi": True,
        "features": "book", "lookback": 50,
        "optimizer": torch.optim.Adam, "lr": 1e-3, "batch_size": 32, "max_epochs": 50,
        "monitor": "loss", "patience": None,
    },
    "tlob": {
        "build": lambda horizons: TLOB(lookback=128), "task": "classification", "multi": False, "features": "book", "lookback": 128,
        "optimizer": torch.optim.Adam, "lr": 1e-4, "ema_decay": 0.999, "batch_size": 128, "max_epochs": 10,
        "monitor": "loss", "patience": 1, "min_delta": 0.002,
    },
    "deepofi_lstm": {
        "build": lambda horizons: DeepOFILSTM(horizons=horizons), "task": "mse", "multi": True, "features": "order_flow", "lookback": 100,
        "optimizer": torch.optim.Adam, "lr": 1e-5, "batch_size": 256, "max_epochs": 50,
        "monitor": "loss", "patience": 5,
    },
    "siamese_lstm": {
        "build": lambda horizons: SiameseLOB(), "task": "mae", "multi": False, "features": "order_flow", "lookback": 50,
        "optimizer": torch.optim.Adam, "lr": 1e-4, "weight_decay": 1e-3, "batch_size": 256, "max_epochs": 50,
        "monitor": "loss", "patience": 5,
    },
    "custom_symmetric_lob": {
        "build": lambda horizons: SymmetricLOB(horizons=horizons), "task": "joint", "multi": True, "features": "custom+context", "lookback": 128,
        "optimizer": torch.optim.AdamW, "lr": 3e-4, "ema_decay": 0.999, "batch_size": 512, "max_epochs": 50,
        "class_weighted": True, "auxiliary_weight": 0.2,
        "monitor": "loss", "patience": 8, "gradient_clip": 1.0, "plateau_patience": 5,
    },
}
CONFIGS = {name: {**DEFAULTS, **config} for name, config in CONFIGS.items()}
