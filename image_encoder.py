"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector

GRID = 7            # 224 / 32 patches along one side
PATCH_SIZE = 32
PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE   # 3072 values per flattened patch


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        # A head owns one contiguous slice of the feature axis: head h reads
        # features [h*dh, (h+1)*dh). reshape only renames that axis, it does not
        # move any value, so (B,N,H,dh) still has the token axis outermost.
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError(f'split_heads expects [B,N,{self.width}], got {x.shape}')
        b, n, _ = x.shape
        # Move the head axis in front of the token axis: [B,N,H,dh] -> [B,H,N,dh].
        # Only then can every head build its own [N,dh] x [dh,N] score matrix.
        return x.reshape(b, n, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        """Inverse of split_heads: [B,H,N,D/H] -> [B,N,D]."""
        b, h, n, dh = x.shape
        # Undo the transpose first, otherwise reshape would splice values of
        # different tokens instead of different heads.
        return x.transpose(0, 2, 1, 3).reshape(b, n, h * dh)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        # TODO-03a: query/key comparisons, scaled by sqrt(head_dim).
        # q,k,v: [B,H,N,64]. scores must have [B,H,N,N].
        # k is 4D, so only its last two axes may be swapped; k.T would reverse
        # all four axes and break the batch/head grouping.
        # The scale is a float32 scalar on purpose: np.sqrt(python_int) is
        # float64 and would silently promote the whole encoder.
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        # TODO-03b: mix value rows using coefficients; then join head outputs.
        # context: [B,H,N,64], merged: [B,N,D].
        context = coefficients @ v
        merged = self.merge_heads(context)

        # TODO-03c: output projection, using out_weight and out_bias.
        # Return [B,N,D]. Remember the stored linear-weight orientation.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition. Name the result residual.
        # The residual base is the block input x, not the normalized copy.
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden: [B,N,3072], mlp_update: [B,N,768] for the image tower.
        # Return residual + mlp_update. Both residual paths retain [B,N,D].
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)                       # exact erf GELU
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias
        return residual + mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f'patchify expects [B,3,224,224], got {images.shape}')
        b = images.shape[0]
        # TODO-01: separate patch-grid axes; reorder; flatten.
        # 224 = 7 * 32, so split H and W into (grid, inside). reshape keeps the
        # memory order, it only names the axes: (B,C,gh,ph,gw,pw).
        grid = images.reshape(b, 3, GRID, PATCH_SIZE, GRID, PATCH_SIZE)
        # Patches are visited row by row, and inside a patch C,H,W stays
        # contiguous: (B,gh,gw,C,ph,pw) is the order conv1 was flattened with.
        patches = grid.transpose(0, 2, 4, 1, 3, 5)
        return patches.reshape(b, GRID * GRID, PATCH_DIM)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project patches to [B,49,768]. There is no patch bias.
        projected = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: prepend CLS on the token axis, then add learned positions.
        # Return [B,50,768]. self.position broadcasts over B.
        tokens = np.concatenate([cls, projected], axis=1)  # (B,50,768)
        return tokens + self.position                      # CLS gets position[0]

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # Select CLS, apply post_norm, and project with self.projection.
        # [B,50,768] -> [B,768] -> [B,512]. visual.proj is already [in,out].
        # Name the projected array output. The final unit-vector helper is given.
        cls_token = tokens[:, 0, :]                         # (B,768)
        normalized = self.post_norm(cls_token)              # feature-axis Norm
        output = normalized @ self.projection               # (B,768)@(768,512)
        return unit_vector(output)
