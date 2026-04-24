import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from VeinKAN.kans.kan import KAN


# =============================================================================
# VEINKAN MODEL DEFINITION
# =============================================================================
class VeinKAN(nn.Module):
    def __init__(self, num_classes=636, pretrained=True, hidden_dim=540):
        """
        VeinKAN model based on "VeinKAN: A Finger Vein Recognition Model
        Based on Kolmogorov-Arnold Networks" (Tran & Tran, 2025).

        Architecture (Paper Section II.B, Fig. 4):
            1. Backbone: InceptionV3 (Pretrained) for Feature Extraction.
            2. Classifier: Deep KAN (Kolmogorov-Arnold Network).

        Paper specifications (Table III, SDUMLA-HMT):
            - Total parameters: 34.81M
            - MMAC: 539.12
            - Inference time: 1.0096 ms

        Args:
            num_classes (int): Number of identities (classes).
                Paper uses 636 (SDUMLA-HMT) or 491 (FV_USM).
            pretrained (bool): Whether to load ImageNet weights for InceptionV3.
            hidden_dim (int): Size of the hidden layer in Deep KAN.
                Default 540 approximates the paper's 34.81M total parameters.
        """
        super(VeinKAN, self).__init__()

        # 1. InceptionV3 Backbone for Feature Extraction
        # Note: pretrained weights require aux_logits=True during loading,
        # so we load with aux_logits and then disable it.
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.backbone = inception_v3(weights=weights, aux_logits=True)
        self.backbone.aux_logits = False
        self.backbone.AuxLogits = None

        # Get feature dimension (2048 for InceptionV3)
        in_features = self.backbone.fc.in_features  # 2048

        # Replace FC with Identity to extract raw features
        self.backbone.fc = nn.Identity()

        # 2. Deep KAN Classifier
        # Paper Eq. 2: KAN(x) = phi_3 o phi_2 o phi_1(x)
        # Structure: [2048] -> [hidden_dim] -> [num_classes]
        self.kan_classifier = KAN(
            layers_hidden=[in_features, hidden_dim, num_classes],
            grid_size=5,
            spline_order=3,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        features = self.backbone(x)             # (B, 2048)
        logits = self.kan_classifier(features)   # (B, num_classes)
        return logits

    def get_embedding(self, x):
        """
        Extract feature embeddings (before classification).

        Returns:
            features: Feature tensor of shape (B, 2048)
        """
        training_state = self.training
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
        self.train(training_state)
        return features


def build_model(num_classes=636, pretrained=True, hidden_dim=540, device='cuda'):
    """
    Build VeinKAN model with optimizer and loss as specified in the paper.

    Paper Section III.B - Experimental Setup:
        - Optimizer: Adam, lr=1e-4
        - Loss: Categorical Cross-Entropy
        - Batch size: 64
        - Early stopping: patience=10, max_epochs=100
        - Input size: 224x224

    Args:
        num_classes (int): Number of classes.
        pretrained (bool): Use ImageNet pretrained weights.
        hidden_dim (int): KAN hidden dimension.
        device (str): Device to use.

    Returns:
        model, optimizer, criterion, config dict
    """
    model = VeinKAN(
        num_classes=num_classes,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    config = {
        'batch_size': 64,
        'max_epochs': 100,
        'early_stopping_patience': 10,
        'input_size': 299,
        'optimizer': 'Adam',
        'lr': 1e-4,
        'loss': 'CrossEntropyLoss',
    }

    return model, optimizer, criterion, config


# =============================================================================
# MAIN TEST
# =============================================================================
if __name__ == "__main__":
    num_classes = 636
    hidden_dim = 540
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model, optimizer, criterion, config = build_model(
            num_classes=num_classes,
            pretrained=False,
            hidden_dim=hidden_dim,
            device=device,
        )

        print("✅ VeinKAN instantiated successfully.")
        print(f"   Structure: InceptionV3 (2048) -> KAN ({hidden_dim}) -> Output ({num_classes})")
        print(f"   Device: {device}")

        # Parameter count (Paper Table III: 34.81M)
        total_params = sum(p.numel() for p in model.parameters())
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        kan_params = sum(p.numel() for p in model.kan_classifier.parameters())
        print(f"\n📊 Parameter Count:")
        print(f"   Backbone (InceptionV3): {backbone_params / 1e6:.2f}M")
        print(f"   KAN Classifier:         {kan_params / 1e6:.2f}M")
        print(f"   Total:                  {total_params / 1e6:.2f}M")
        print(f"   Paper target:           34.81M")

        # Training config
        print(f"\n⚙️  Training Config:")
        for k, v in config.items():
            print(f"   {k}: {v}")
        print(f"   Optimizer: {optimizer.__class__.__name__} (lr={optimizer.defaults['lr']})")
        print(f"   Loss: {criterion.__class__.__name__}")

        # Forward pass test
        x = torch.randn(2, 3, 299, 299).to(device)
        model.eval()
        out = model(x)

        print(f"\n🔄 Forward Pass:")
        print(f"   Input:  {x.shape}")
        print(f"   Output: {out.shape}")

        if out.shape == (2, num_classes):
            print("✅ Output shape correct.")
        else:
            print("❌ Output shape incorrect.")

        # Test loss computation
        labels = torch.randint(0, num_classes, (2,)).to(device)
        loss = criterion(out, labels)
        print(f"\n📉 Loss test: {loss.item():.4f}")
        print("✅ Loss computation works.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()