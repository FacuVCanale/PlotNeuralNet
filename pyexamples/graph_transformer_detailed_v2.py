import sys
sys.path.append('../')
from pycore.tikzeng import *

# =============================================================
# GraphTransformer Architecture Diagram - DETAILED v2
# -------------------------------------------------------------
# This script creates a comprehensive TikZ/LaTeX figure showing:
# - Complete GraphTransformer architecture overview
# - Detailed zoom into TransformerConv internal steps
# - Node-wise graph attention mechanics
# - Tensor shapes and dimensions
# =============================================================

# Architecture parameters from your GraphTransformer
NUM_NODES = 522
NUM_FEATURES = 22 
HIDDEN_DIM = 64
NUM_HEADS = 2
D_HEAD = HIDDEN_DIM // NUM_HEADS  # 32
NUM_LAYERS = 2
NUM_TARGETS = 1
DROPOUT_RATE = 0.05

# Extended component functions for detailed visualization
def to_Dropout(name, rate=0.2, offset="(0,0,0)", to="(0,0,0)", width=0.5, height=10, depth=10):
    """Create a Dropout component"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Dropout,
        xlabel={{" ","dummy"}},
        fill=\PoolColor,
        opacity=0.4,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_BatchNorm(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=15, depth=15):
    """Create a BatchNorm component"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=BatchNorm,
        fill=\FcColor,
        opacity=0.6,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_GraphInput(name, offset="(0,0,0)", to="(0,0,0)", width=4, height=35, depth=35):
    """Create graph input representation"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Graph Input,
        xlabel={{"""+ str(NUM_NODES) +r""",}},
        zlabel=""" + str(NUM_FEATURES) + r""",
        fill=\ConvColor,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_NodeWiseLinear(name, operation, in_dim, out_dim, offset="(0,0,0)", to="(0,0,0)", width=2, height=20, depth=15):
    """Create node-wise linear transformation"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=""" + operation + r""",
        xlabel={{"""+ str(in_dim) +r""",}},
        zlabel=""" + str(out_dim) + r""",
        fill=\FcColor,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_QKVProjections(name_prefix, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=25, depth=20):
    """Create Q, K, V projections for each head"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name_prefix + r""",
        caption=Q K V Projections,
        xlabel={{"Query","Key","Value"}},
        zlabel="shape",
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +r""",
        width={""" + str(width) + "," + str(width) + "," + str(width) + r"""},
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_GraphAttention(name, offset="(0,0,0)", to="(0,0,0)", width=4, height=25, depth=25):
    """Create graph attention computation"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Graph Attention,
        xlabel={{"edges", "heads"}},
        fill=\ConvReluColor,
        opacity=0.8,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_MessageAggregation(name, offset="(0,0,0)", to="(0,0,0)", width=3, height=22, depth=22):
    """Create message aggregation step"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Message Aggregation,
        xlabel={{"nodes", "heads", "dims"}},
        fill=\SumColor,
        opacity=0.7,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_HeadFusion(name, fusion_type="Average", offset="(0,0,0)", to="(0,0,0)", width=2.5, height=20, depth=20):
    """Create head fusion step"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=""" + fusion_type + r""" Heads,
        xlabel={{"nodes", "64"}},
        fill=\SumColor,
        opacity=0.8,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_ResidualConnection(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5):
    """Create residual connection with skip path"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Ball={
        name=""" + name +r""",
        fill=\SumColor,
        opacity=0.9,
        radius="""+ str(radius) +r""",
        logo=$+$
        }
    };
"""

def to_BatchNormGELU(name, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=18, depth=18):
    """Create combined BatchNorm + GELU block"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name + r""",
        caption=BN + GELU,
        xlabel={{"BN","GELU"}},
        fill=\FcColor,
        bandfill=\FcReluColor,
        height="""+ str(height) +r""",
        width={1,1},
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_MLPPredictor(name, offset="(0,0,0)", to="(0,0,0)", width=3, height=25, depth=15):
    """Create MLP predictor block"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name + r""",
        caption=MLP Predictor,
        xlabel={{"64","32","1"}},
        fill=\FcColor,
        bandfill=\FcReluColor,
        height="""+ str(height) +r""",
        width={2,1.5,1},
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_FinalOutput(name, offset="(0,0,0)", to="(0,0,0)", width=2, height=35, depth=8):
    """Create final output"""
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Demand Prediction,
        xlabel={{"522", "1"}},
        fill=\SoftmaxColor,
        opacity=0.9,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

# Build the comprehensive architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
]

# Add title annotation
arch.append(r"""
\node[anchor=north, align=center, font=\Large\bfseries] at (15, 8, 0) 
    {Graph Transformer Architecture v2 \\ 
     {\footnotesize Detailed TransformerConv Internal Mechanics}};
""")

# ===== MAIN ARCHITECTURE FLOW =====

# 1. Graph Input
arch.append(
    to_GraphInput(
        "graph_input",
        offset="(0,0,0)",
        to="(0,0,0)",
        width=4,
        height=35,
        depth=35,
    )
)

# 2. First TransformerConv Layer - DETAILED BREAKDOWN
prev_layer = "graph_input"

# Step 1: Q, K, V Projections
qkv1_name = "qkv1_detailed"
arch.append(
    to_QKVProjections(
        qkv1_name,
        offset="(4,0,0)",
        to=f"({prev_layer}-east)",
        width=1.5,
        height=28,
        depth=25,
    )
)
arch.append(to_connection(prev_layer, qkv1_name))

# Step 2: Graph Attention Computation  
attn1_name = "graph_attn1"
arch.append(
    to_GraphAttention(
        attn1_name,
        offset="(3,0,0)",
        to=f"({qkv1_name}-east)",
        width=4,
        height=25,
        depth=25,
    )
)
arch.append(to_connection(qkv1_name, attn1_name))

# Step 3: Message Aggregation
msg1_name = "msg_agg1"
arch.append(
    to_MessageAggregation(
        msg1_name,
        offset="(3,0,0)",
        to=f"({attn1_name}-east)",
        width=3,
        height=22,
        depth=22,
    )
)
arch.append(to_connection(attn1_name, msg1_name))

# Step 4: Head Fusion (Averaging)
fusion1_name = "head_fusion1"
arch.append(
    to_HeadFusion(
        fusion1_name,
        "Average",
        offset="(2.5,0,0)",
        to=f"({msg1_name}-east)",
        width=2.5,
        height=20,
        depth=20,
    )
)
arch.append(to_connection(msg1_name, fusion1_name))

# Step 5: Residual Connection
residual1_name = "residual1"
arch.append(
    to_ResidualConnection(
        residual1_name,
        offset="(2,0,0)",
        to=f"({fusion1_name}-east)",
        radius=2.5,
    )
)
arch.append(to_connection(fusion1_name, residual1_name))

# Skip connection from input (simplified representation)
arch.append(to_skip("graph_input", residual1_name, pos=1.25))

# Step 6: BatchNorm + GELU + Dropout
bn_gelu1_name = "bn_gelu1"
arch.append(
    to_BatchNormGELU(
        bn_gelu1_name,
        offset="(2.5,0,0)",
        to=f"({residual1_name}-east)",
        width=1.5,
        height=18,
        depth=18,
    )
)
arch.append(to_connection(residual1_name, bn_gelu1_name))

# Add dropout visualization
dropout1_name = "dropout1"
arch.append(
    to_Dropout(
        dropout1_name,
        DROPOUT_RATE,
        offset="(1.5,0,0)",
        to=f"({bn_gelu1_name}-east)",
        width=0.8,
        height=15,
        depth=15,
    )
)
arch.append(to_connection(bn_gelu1_name, dropout1_name))

# ===== SECOND TRANSFORMERCONV LAYER (Simplified) =====
if NUM_LAYERS > 1:
    prev_layer = dropout1_name
    
    # Second layer - more compact representation
    tconv2_name = "tconv2_compact"
    arch.append(
        to_Conv(
            tconv2_name,
            s_filer=HIDDEN_DIM,
            n_filer=HIDDEN_DIM,
            offset="(3,0,0)",
            to=f"({prev_layer}-east)",
            height=20,
            width=4,
            depth=20,
            caption="TransformerConv Layer 2",
        )
    )
    arch.append(to_connection(prev_layer, tconv2_name))
    
    # BatchNorm only for final layer
    bn_final_name = "bn_final"
    arch.append(
        to_BatchNorm(
            bn_final_name,
            offset="(2,0,0)",
            to=f"({tconv2_name}-east)",
            width=1,
            height=18,
            depth=18,
        )
    )
    arch.append(to_connection(tconv2_name, bn_final_name))
    
    final_conv_layer = bn_final_name
else:
    final_conv_layer = dropout1_name

# ===== MLP PREDICTOR =====
mlp_name = "mlp_predictor"
arch.append(
    to_MLPPredictor(
        mlp_name,
        offset="(3,0,0)",
        to=f"({final_conv_layer}-east)",
        width=3,
        height=22,
        depth=12,
    )
)
arch.append(to_connection(final_conv_layer, mlp_name))

# ===== FINAL OUTPUT =====
arch.append(
    to_FinalOutput(
        "final_output",
        offset="(3,0,0)",
        to=f"({mlp_name}-east)",
        width=2,
        height=30,
        depth=8,
    )
)
arch.append(to_connection(mlp_name, "final_output"))

# Add detailed annotations
arch.append(r"""
\node[anchor=north west, align=left, font=\footnotesize] at (-2, -5, 0) 
    {{\bf TransformerConv Internal Steps:} \\
     1. Node-wise Q,K,V projections: R22 to R64 \\\\
     2. Graph attention: alpha = softmax(Q*K/sqrt(d)) \\\\
     3. Message aggregation: m = sum(alpha*V) \\\\
     4. Head fusion: Average (concat=False) \\\\
     5. Residual: h = m + W*x \\\\
     6. Normalization and activation};
""")

arch.append(r"""
\node[anchor=north west, align=left, font=\footnotesize] at (20, -5, 0) 
    {{\bf Tensor Shapes:} \\
     Input: x in R(522x22) \\\\
     Q,K,V: in R(522x2x32) \\\\
     Attention: alpha \\\\
     Messages: m in R(522x64) \\\\
     Output: y hat in R(522x1)};
""")

# Finish
arch.append(to_end())

def main():
    namefile = str(sys.argv[1]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main() 