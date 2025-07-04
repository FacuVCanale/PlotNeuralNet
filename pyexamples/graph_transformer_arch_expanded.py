import sys
sys.path.append('../')
from pycore.tikzeng import *

# =============================================================
# GraphTransformer Architecture Diagram - EXPANDED VERSION
# -------------------------------------------------------------
# This script creates a detailed TikZ/LaTeX figure that visualizes
# the internal structure of the GraphTransformer model, showing:
# - TransformerConv layers with their components
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

def to_TransformerConv(name, in_dim, out_dim, heads, offset="(0,0,0)", to="(0,0,0)", width=4, height=25, depth=25):
    """Create a TransformerConv layer with detailed caption"""
    caption = f"T.Conv {heads} head{'' if heads == 1 else 's'}\\\\[-25cm]~"
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(in_dim) +r""",}},
        zlabel="""+ str(out_dim) +r""",
        fill=\ConvColor,
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

# Layer 1: Input -> Hidden with multi-head attention
prev_layer = "input"
layer_name = "trconv1"
arch.append(
    to_TransformerConv(
        layer_name,
        NUM_FEATURES,
        HIDDEN_DIM,
        NUM_HEADS,
        offset="(3,0,0)",
        to=f"({prev_layer}-east)",
        width=4,
        height=30,
        depth=30,
    )
)
arch.append(to_connection(prev_layer, layer_name))

# BatchNorm after Layer 1
bn_name = "bn1"
arch.append(
    to_BatchNorm(
        bn_name,
        offset="(1.5,0,0)",
        to=f"({layer_name}-east)",
        width=1,
        height=25,
        depth=25,
    )
)
arch.append(to_connection(layer_name, bn_name))

# GELU activation after BatchNorm
gelu_name = "gelu1"
arch.append(
    to_Activation(
        gelu_name,
        "GELU",
        offset="(1.2,0,0)",
        to=f"({bn_name}-east)",
        width=1,
        height=20,
        depth=20,
    )
)
arch.append(to_connection(bn_name, gelu_name))

# Dropout after GELU
dropout_name = "dropout1"
arch.append(
    to_Dropout(
        dropout_name,
        DROPOUT_RATE,
        offset="(1.0,0,0)",
        to=f"({gelu_name}-east)",
        width=0.5,
        height=15,
        depth=15,
    )
)
arch.append(to_connection(gelu_name, dropout_name))

# Layer 2: Hidden -> Hidden (if NUM_LAYERS > 2)
if NUM_LAYERS > 2:
    prev_layer = dropout_name
    layer_name = "trconv2"
    arch.append(
        to_TransformerConv(
            layer_name,
            HIDDEN_DIM,
            HIDDEN_DIM,
            NUM_HEADS,
            offset="(3,0,0)",
            to=f"({prev_layer}-east)",
            width=4,
            height=25,
            depth=25,
        )
    )
    arch.append(to_connection(prev_layer, layer_name))

    # BatchNorm after Layer 2
    bn_name = "bn2"
    arch.append(
        to_BatchNorm(
            bn_name,
            offset="(1.5,0,0)",
            to=f"({layer_name}-east)",
            width=1,
            height=20,
            depth=20,
        )
    )
    arch.append(to_connection(layer_name, bn_name))

    # GELU activation after BatchNorm
    gelu_name = "gelu2"
    arch.append(
        to_Activation(
            gelu_name,
            "GELU",
            offset="(1.2,0,0)",
            to=f"({bn_name}-east)",
            width=1,
            height=18,
            depth=18,
        )
    )
    arch.append(to_connection(bn_name, gelu_name))

    # Dropout after GELU
    dropout_name = "dropout2"
    arch.append(
        to_Dropout(
            dropout_name,
            DROPOUT_RATE,
            offset="(1.0,0,0)",
            to=f"({gelu_name}-east)",
            width=0.5,
            height=15,
            depth=15,
        )
    )
    arch.append(to_connection(gelu_name, dropout_name))

# Final TransformerConv layer (single head)
prev_layer = dropout_name
layer_name = "trconv_final"
arch.append(
    to_TransformerConv(
        layer_name,
        HIDDEN_DIM,
        HIDDEN_DIM,
        1,  # Single head for final layer
        offset="(3,0,0)",
        to=f"({prev_layer}-east)",
        width=4,
        height=25,
        depth=25,
    )
)
arch.append(to_connection(prev_layer, layer_name))

# BatchNorm after final TransformerConv
bn_final_name = "bn_final"
arch.append(
    to_BatchNorm(
        bn_final_name,
        offset="(1.5,0,0)",
        to=f"({layer_name}-east)",
        width=1,
        height=20,
        depth=20,
    )
)
arch.append(to_connection(layer_name, bn_final_name))

# MLP Predictor - First Linear Layer
mlp1_name = "mlp1"
arch.append(
    to_Linear(
        mlp1_name,
        HIDDEN_DIM,
        HIDDEN_DIM // 2,
        offset="(3,0,0)",
        to=f"({bn_final_name}-east)",
        width=2.5,
        height=20,
        depth=15,
    )
)
arch.append(to_connection(bn_final_name, mlp1_name))

# GELU activation in MLP
mlp_gelu_name = "mlp_gelu"
arch.append(
    to_Activation(
        mlp_gelu_name,
        "GELU",
        offset="(1.2,0,0)",
        to=f"({mlp1_name}-east)",
        width=1,
        height=15,
        depth=12,
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
        height=12,
        depth=10,
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
        height=15,
        depth=8,
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
        height=40,
        depth=8,
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