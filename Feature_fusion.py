import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        visual_dim=779,
        text_dim=384,
        hidden_dim=512,
        num_heads=8
    ):
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_features, text_embedding):
        """
        visual_features shape: [batch_size, num_frames, 779]
        text_embedding shape:  [batch_size, 384]

        output shape: [batch_size, num_frames, 512]
        """

        visual_tokens = self.visual_proj(visual_features)

        text_token = self.text_proj(text_embedding)

        # Convert text from [B, 512] to [B, 1, 512]
        text_token = text_token.unsqueeze(1)

        # Visual attends to text
        attended_visual, attention_weights = self.cross_attention(
            query=visual_tokens,
            key=text_token,
            value=text_token
        )

        fused = self.norm(visual_tokens + attended_visual)

        ff_output = self.feed_forward(fused)

        fused = self.output_norm(fused + ff_output)

        return fused, attention_weights