import sys
sys.path.append('../')
from pycore.tikzeng import *

# =============================================================
# GraphTransformer Architecture Diagram - EXPANDED ATTENTION VERSION
# -------------------------------------------------------------
# This script creates a detailed TikZ/LaTeX figure that visualizes
# the internal structure of the GraphTransformer model, showing:
# - Expanded multi-head attention with Q, K, V components
# - Attention matrix computation and averaging (concat=False)
# - BatchNorm, GELU, and Dropout operations
# - Detailed MLP predictor breakdown
# =============================================================

# Architecture parameters matching your GTN class
NUM_NODES = 522      # input features per node
NUM_FEATURES = 22          # number of graph nodes
HIDDEN_DIM = 64         # hidden dimension (d_model)
NUM_HEADS = 2           # attention heads (matching your GTN default)
NUM_LAYERS = 2          # total TransformerConv layers (matching your GTN default)
NUM_TARGETS = 1         # output targets (matching your GTN default)
DROPOUT_RATE = 0.05      # dropout rate

# Custom component functions
def to_BatchNorm(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=15, depth=15):
    """Create a BatchNorm component (no size/xlabel)"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption=BatchNorm,
        fill=\FcColor,
        opacity=0.6,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Activation(name, activation_type="GELU", offset="(0,0,0)", to="(0,0,0)", width=1, height=12, depth=12):
    """Create an activation function component"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ activation_type +r""",
        xlabel={{" ","dummy"}},
        fill=\FcReluColor,
        opacity=0.8,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Dropout(name, rate=0.2, offset="(0,0,0)", to="(0,0,0)", width=0.5, height=10, depth=10):
    """Create a Dropout component"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption=Dropout\\\\""" + str(rate) + r""",
        xlabel={{" ","dummy"}},
        fill=\PoolColor,
        opacity=0.4,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Linear(name, in_dim, out_dim, offset="(0,0,0)", to="(0,0,0)", width=2, height=20, depth=15):
    """Create a Linear layer"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption=Linear\\\\""" + str(in_dim) + "x" + str(out_dim) + r""",
        xlabel={{"""+ str(in_dim) +r""",}},
        zlabel="""+ str(out_dim) +r""",
        fill=\FcColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Output(name, s_filer=10, n_filer=" ", offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{""" + str(n_filer) + r""", "dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Expanded Attention Components
def to_QKV_Transform(name_prefix, in_dim, out_dim, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=18, depth=18):
    """Create Q, K, V transformation blocks using RightBandedBox"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {RightBandedBox={
        name=""" + name_prefix + r""",
        caption=Q K V\\Transform,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width={"""+ str(width) +""","""+ str(width) +""","""+ str(width) +"""},
        depth="""+ str(depth) +"""
        }
    };
"""

def to_MultiHeadAttention(name, heads, offset="(0,0,0)", to="(0,0,0)", width=3, height=22, depth=22):
    """Create multi-head attention computation block"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {RightBandedBox={
        name=""" + name +r""",
        caption=Multi-Head\\Attention,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +""",
        width={""" + ",".join([str(width//heads + 0.5) for _ in range(heads)]) + r"""},
        depth="""+ str(depth) +"""
        }
    };
"""

def to_AttentionAverage(name, heads, offset="(0,0,0)", to="(0,0,0)", width=2, height=18, depth=18):
    """Create attention averaging block"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption=Average\\\\ """ + str(heads) + r""" heads,
        xlabel={{" ","dummy"}},
        fill=\SumColor,
        opacity=0.7,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_ResidualAdd(name, offset="(0,0,0)", to="(0,0,0)", radius=2):
    """Create residual addition component"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Ball={
        name=""" + name +""",
        fill=\SumColor,
        opacity=0.8,
        radius="""+ str(radius) +""",
        logo=$+$
        }
    };
"""

# Build the architecture list
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
]

# Input layer
arch.append(
    to_Conv(
        "input",
        s_filer=NUM_FEATURES,
        n_filer=NUM_NODES,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=50,
        width=6,
        depth=50,
        caption=f"Station Graph\\\\[0pt]{NUM_NODES} nodes\\\\[0pt]{NUM_FEATURES} features\\\\[-5cm]~",
    )
)

# Layer 1: Expanded Multi-Head Attention
prev_layer = "input"

# Q, K, V Transformations
qkv_name = "qkv1"
arch.append(
    to_QKV_Transform(
        qkv_name,
        NUM_FEATURES,
        HIDDEN_DIM,
        offset="(3,0,0)",
        to=f"({prev_layer}-east)",
        width=1.5,
        height=25,
        depth=25,
    )
)
arch.append(to_connection(prev_layer, qkv_name))

# Multi-Head Attention Computation
mha_name = "mha1"
arch.append(
    to_MultiHeadAttention(
        mha_name,
        NUM_HEADS,
        offset="(2,0,0)",
        to=f"({qkv_name}-east)",
        width=3,
        height=22,
        depth=22,
    )
)
arch.append(to_connection(qkv_name, mha_name))

# Attention Averaging
avg_name = "avg1"
arch.append(
    to_AttentionAverage(
        avg_name,
        NUM_HEADS,
        offset="(2,0,0)",
        to=f"({mha_name}-east)",
        width=2,
        height=18,
        depth=18,
    )
)
arch.append(to_connection(mha_name, avg_name))

# Output Projection
proj_name = "proj1"
arch.append(
    to_Linear(
        proj_name,
        HIDDEN_DIM,
        HIDDEN_DIM,
        offset="(2,0,0)",
        to=f"({avg_name}-east)",
        width=2,
        height=18,
        depth=15,
    )
)
arch.append(to_connection(avg_name, proj_name))

# Residual Connection
residual_name = "residual1"
arch.append(
    to_ResidualAdd(
        residual_name,
        offset="(1.5,0,0)",
        to=f"({proj_name}-east)",
        radius=2,
    )
)
arch.append(to_connection(proj_name, residual_name))

# Skip connection from input (simplified representation)
arch.append(to_skip("input", residual_name, pos=1.25))

# BatchNorm after attention
bn_name = "bn1"
arch.append(
    to_BatchNorm(
        bn_name,
        offset="(2,0,0)",
        to=f"({residual_name}-east)",
        width=1,
        height=18,
        depth=18,
    )
)
arch.append(to_connection(residual_name, bn_name))

# GELU activation
gelu_name = "gelu1"
arch.append(
    to_Activation(
        gelu_name,
        "GELU",
        offset="(1.2,0,0)",
        to=f"({bn_name}-east)",
        width=1,
        height=15,
        depth=15,
    )
)
arch.append(to_connection(bn_name, gelu_name))

# Dropout
dropout_name = "dropout1"
arch.append(
    to_Dropout(
        dropout_name,
        DROPOUT_RATE,
        offset="(1.0,0,0)",
        to=f"({gelu_name}-east)",
        width=0.5,
        height=12,
        depth=12,
    )
)
arch.append(to_connection(gelu_name, dropout_name))

# Layer 2: Second Expanded Multi-Head Attention (if NUM_LAYERS > 1)
if NUM_LAYERS > 1:
    prev_layer = dropout_name
    
    # Q, K, V Transformations for layer 2
    qkv2_name = "qkv2"
    arch.append(
        to_QKV_Transform(
            qkv2_name,
            HIDDEN_DIM,
            HIDDEN_DIM,
            offset="(3,0,0)",
            to=f"({prev_layer}-east)",
            width=1.5,
            height=20,
            depth=20,
        )
    )
    arch.append(to_connection(prev_layer, qkv2_name))

    # Multi-Head Attention Computation for layer 2
    mha2_name = "mha2"
    arch.append(
        to_MultiHeadAttention(
            mha2_name,
            NUM_HEADS,
            offset="(2,0,0)",
            to=f"({qkv2_name}-east)",
            width=3,
            height=18,
            depth=18,
        )
    )
    arch.append(to_connection(qkv2_name, mha2_name))

    # Attention Averaging for layer 2
    avg2_name = "avg2"
    arch.append(
        to_AttentionAverage(
            avg2_name,
            NUM_HEADS,
            offset="(2,0,0)",
            to=f"({mha2_name}-east)",
            width=2,
            height=15,
            depth=15,
        )
    )
    arch.append(to_connection(mha2_name, avg2_name))

    # Output Projection for layer 2
    proj2_name = "proj2"
    arch.append(
        to_Linear(
            proj2_name,
            HIDDEN_DIM,
            HIDDEN_DIM,
            offset="(2,0,0)",
            to=f"({avg2_name}-east)",
            width=2,
            height=15,
            depth=12,
        )
    )
    arch.append(to_connection(avg2_name, proj2_name))

    # Residual Connection for layer 2
    residual2_name = "residual2"
    arch.append(
        to_ResidualAdd(
            residual2_name,
            offset="(1.5,0,0)",
            to=f"({proj2_name}-east)",
            radius=1.8,
        )
    )
    arch.append(to_connection(proj2_name, residual2_name))

    # Skip connection from previous layer
    arch.append(to_skip(dropout_name, residual2_name, pos=1.25))

    # BatchNorm after layer 2 attention
    bn2_name = "bn2"
    arch.append(
        to_BatchNorm(
            bn2_name,
            offset="(2,0,0)",
            to=f"({residual2_name}-east)",
            width=1,
            height=15,
            depth=15,
        )
    )
    arch.append(to_connection(residual2_name, bn2_name))

    # GELU activation for layer 2
    gelu2_name = "gelu2"
    arch.append(
        to_Activation(
            gelu2_name,
            "GELU",
            offset="(1.2,0,0)",
            to=f"({bn2_name}-east)",
            width=1,
            height=12,
            depth=12,
        )
    )
    arch.append(to_connection(bn2_name, gelu2_name))

    # Dropout for layer 2
    dropout2_name = "dropout2"
    arch.append(
        to_Dropout(
            dropout2_name,
            DROPOUT_RATE,
            offset="(1.0,0,0)",
            to=f"({gelu2_name}-east)",
            width=0.5,
            height=10,
            depth=10,
        )
    )
    arch.append(to_connection(gelu2_name, dropout2_name))
    
    final_layer = dropout2_name
else:
    final_layer = dropout_name

# MLP Predictor - First Linear Layer
mlp1_name = "mlp1"
arch.append(
    to_Linear(
        mlp1_name,
        HIDDEN_DIM,
        HIDDEN_DIM // 2,
        offset="(3,0,0)",
        to=f"({final_layer}-east)",
        width=2.5,
        height=15,
        depth=12,
    )
)
arch.append(to_connection(final_layer, mlp1_name))

# GELU activation in MLP
mlp_gelu_name = "mlp_gelu"
arch.append(
    to_Activation(
        mlp_gelu_name,
        "GELU",
        offset="(1.2,0,0)",
        to=f"({mlp1_name}-east)",
        width=1,
        height=12,
        depth=10,
    )
)
arch.append(to_connection(mlp1_name, mlp_gelu_name))

# Dropout in MLP
mlp_dropout_name = "mlp_dropout"
arch.append(
    to_Dropout(
        mlp_dropout_name,
        DROPOUT_RATE,
        offset="(1.0,0,0)",
        to=f"({mlp_gelu_name}-east)",
        width=0.5,
        height=10,
        depth=8,
    )
)
arch.append(to_connection(mlp_gelu_name, mlp_dropout_name))

# Final Linear Layer
mlp2_name = "mlp2"
arch.append(
    to_Linear(
        mlp2_name,
        HIDDEN_DIM // 2,
        NUM_TARGETS,
        offset="(2,0,0)",
        to=f"({mlp_dropout_name}-east)",
        width=2,
        height=12,
        depth=6,
    )
)
arch.append(to_connection(mlp_dropout_name, mlp2_name))

# Output prediction
arch.append(
    to_Output(
        "output",
        NUM_TARGETS,
        n_filer=NUM_NODES,
        offset="(2,0,0)",
        to=f"({mlp2_name}-east)",
        width=2,
        height=35,
        depth=6,
        caption="Arrivals",
    )
)
arch.append(to_connection(mlp2_name, "output"))

# Finish
arch.append(to_end())

def main():
    namefile = str(sys.argv[1]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main() 