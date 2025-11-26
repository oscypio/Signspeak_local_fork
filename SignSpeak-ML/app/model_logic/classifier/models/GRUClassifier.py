import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """
    Runtime-only GRU classifier for ASL sign recognition.
    Loads a fully packaged model from file and supports only inference.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        proj_dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes

        # === GRU core ===
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # === Projection head ===
        last_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(last_dim),
            nn.Dropout(proj_dropout),
            nn.Linear(last_dim, num_classes)
        )

    # ---------------------------------------------------------
    # FORWARD: x = (B,T,F)
    # ---------------------------------------------------------
    def forward(self, x):
        """
        Returns logits: (B, num_classes)
        """

        _, h_n = self.gru(x)

        # last hidden state
        if self.gru.bidirectional:
            h_fwd, h_bwd = h_n[-2], h_n[-1]
            last_hidden = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            last_hidden = h_n[-1]

        logits = self.head(last_hidden)
        return logits

    # ---------------------------------------------------------
    # PREDICT: returns class index
    # ---------------------------------------------------------
    def predict(self, x):
        """
        x: (1,T,F)
        lengths: (1,)
        Returns: predicted class index (int)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1).item()

    # ---------------------------------------------------------
    # PREDICT WITH PROBABILITIES
    # ---------------------------------------------------------
    def predict_proba(self, x, lengths):
        """
        Returns dictionary: {class_idx: probability}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)[0]
            return {i: float(p) for i, p in enumerate(probs)}

    # ---------------------------------------------------------
    # LOAD MODEL FROM BLACKBOX PACKAGE
    # ---------------------------------------------------------
    @classmethod
    def from_file(cls, path: str, device: str = "cpu"):
        package = torch.load(path, map_location=device)

        # ----------------------------------------------------------
        # Support for both formats:
        #
        #   { state_dict, meta, model_kwargs, normalization }
        #   { state_dict, config: {meta, model_kwargs}, normalization }
        # ----------------------------------------------------------

        # New format (your version now)
        if "config" in package:
            meta = package["config"]["meta"]
            model_kwargs = package["config"]["model_kwargs"]

        # Old flat format
        else:
            meta = package["meta"]
            model_kwargs = package["model_kwargs"]

        state_dict = package["state_dict"]

        # Build model
        model = cls(**model_kwargs)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # attach metadata
        model.class_names = meta.get("class_names")
        model.class_to_idx = meta.get("class_to_idx")
        model.input_size = meta.get("input_size")

        return model

