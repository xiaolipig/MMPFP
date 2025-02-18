import mindspore.nn as nn
import mindspore.ops as ops
from biomodel.A_part import ProteinTransformer
from biomodel.B_part import FeatureFusion


class MultiHeadSelfAttention(nn.Cell):

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.out_proj = nn.Dense(embed_dim, embed_dim)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        scores = ops.BatchMatMul()(Q, K.swapaxes(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = self.softmax(scores)
        attn_output = ops.BatchMatMul()(attn_weights, V)
        attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        return output

# final model
class FinalProteinModel(nn.Cell):


    def __init__(self, protein_dim, fusion_dim, output_dim, num_heads=8, num_layers=4):
        super(FinalProteinModel, self).__init__()
        self.protein_transformer = ProteinTransformer(protein_dim, num_heads, num_layers)
        self.feature_fusion = FeatureFusion(fusion_dim, fusion_dim, fusion_dim)
        self.protein_fc = nn.Dense(protein_dim, fusion_dim)
        self.mhsa = MultiHeadSelfAttention(fusion_dim * 2, num_heads)
        self.output_fc = nn.Dense(fusion_dim * 2, output_dim)
        self.relu = nn.ReLU()

    def construct(self, protein_features, gcn_features, repvgg_features):
        protein_features = self.relu(self.protein_fc(self.protein_transformer(protein_features)))
        fusion_features = self.feature_fusion(gcn_features, repvgg_features)
        combined_features = ops.Concat(1)((protein_features, fusion_features))
        attn_output = self.mhsa(combined_features.unsqueeze(1)).squeeze(1)
        output = self.output_fc(attn_output)
        return output