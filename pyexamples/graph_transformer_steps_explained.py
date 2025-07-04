import sys
sys.path.append('../')
from pycore.tikzeng import *

# =============================================================================
# GraphTransformer with Edge Features - COMPLETE STEP-BY-STEP IMPLEMENTATION
# =============================================================================
# 
# This script implements ALL STEPS from the detailed tensor table:
#
# | Paso | Tensor / operación                          | Dimensión correcta      | Notas clave                             |
# | ---- | ------------------------------------------- | ----------------------- | --------------------------------------- |
# | 0    | **x** (features nodos)                      | **(522 × 22)**          | —                                       |
# | 0′   | **edge\_index**                             | **(2 × 5)**             | Lista de 5 aristas (origen, destino).   |
# | 0″   | **edge\_attr** *(opcional)*                 | **(5 × d_e)**           | Si existen atributos por arista.        |
# | 1    | Q / K / V capa 1 → proyección + **reshape** | **(522 × 2 × 64)**      | 2 cabezas, 64 dims cada una.            |
# | 1′   | edge\_attr → W₆ → **reshape**               | **(5 × 2 × 64)**        | Se suma a K_j, V_j.                     |
# | 2    | **α₁** coeficientes de atención             | **(5 × 2)**             | Por arista y cabeza.                    |
# | 2′   | **m₁** mensajes agregados (Σ α·V)           | **(522 × 2 × 64)**      | Antes de fusionar cabezas.              |
# | 3    | **Promedio cabezas** (concat=False)         | **(522 × 64)**          | —                                       |
# | 3′   | **r₁** skip connection (W₁·x)               | **(522 × 64)**          | Residual directo del nodo.              |
# | 3″   | **h₁** = r₁ + m₁ (β desactivado)            | **(522 × 64)**          | **← Residual aplicado aquí**            |
# | 4    | BatchNorm 1 → GELU → Dropout                | **(522 × 64)**          | —                                       |
# | 5    | Q / K / V capa 2 → proyección + **reshape** | **(522 × 1 × 64)**      | 1 cabeza.                               |
# | 5′   | edge\_attr → W₆ → **reshape**               | **(5 × 1 × 64)**        | Idem capa 1.                            |
# | 6    | **α₂** coeficientes de atención             | **(5 × 1)**             | —                                       |
# | 6′   | **m₂** mensajes agregados                   | **(522 × 1 × 64)**      | —                                       |
# | 7    | **Promedio cabezas**                        | **(522 × 64)**          | —                                       |
# | 7′   | **r₂** skip connection (W₁·h₁)              | **(522 × 64)**          | —                                       |
# | 7″   | **h₂** = r₂ + m₂ (β desactivado)            | **(522 × 64)**          | **← Segundo residual**                  |
# | 8    | BatchNorm 2                                 | **(522 × 64)**          | —                                       |
# | 9    | Linear 64 → 32                              | **(522 × 32)**          | —                                       |
# | 10   | Linear 32 → 1 (ŷ)                           | **(522 × 1)**           | Predicción final por nodo.              |
#
# =============================================================================

# Architecture parameters exactly matching the table
NUM_NODES = 522
NUM_FEATURES = 22
NUM_EDGES = 5
EDGE_DIM = 64  # d_e in the table
HIDDEN_DIM = 64
NUM_HEADS_L1 = 2  # Layer 1: 2 heads
NUM_HEADS_L2 = 1  # Layer 2: 1 head (reduced)
D_HEAD_L1 = HIDDEN_DIM  # Duplicated
D_HEAD_L2 = HIDDEN_DIM  # 64 per head
NUM_TARGETS = 1
DROPOUT_RATE = 0.05

# Diagram layout parameters
HORIZONTAL_SPACING = 2  # Horizontal spacing between components (was hardcoded as (3,0,0))

def to_extended_colors():
    return r"""\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,2;green,5;white,5}
\def\FcReluColor{rgb:blue,2;green,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\EdgeColor{rgb:green,8;blue,2;white,3}
\def\EdgeReluColor{rgb:green,8;blue,5;white,2}
\def\AttentionColor{rgb:orange,6;red,3;white,3}
\def\GraphColor{rgb:purple,5;blue,3;white,4}
"""

def to_EdgeIndex(name, num_edges, offset="(0,0,0)", to="(0,0,0)", width=1, height=8, depth=15):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Edge Indexes,
        zlabel="""+ str(num_edges) +r""",
        fill=\GraphColor,
        opacity=0.7,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_EdgeInput(name, num_edges, edge_dim, offset="(0,0,0)", to="(0,0,0)", width=2, height=15, depth=25):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Edge Features,
        zlabel="""+ str(edge_dim) +r""",
        fill=\EdgeColor,
        opacity=0.8,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_QKV_WithEdges(name_prefix, in_dim, out_dim, heads, offset="(0,0,0)", to="(0,0,0)", width=2, height=25, depth=20):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name_prefix + r""",
        caption=QKV Transform,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""+ str(height) +r""",
        width={"""+ str(width) +r""","""+ str(width) +r""","""+ str(width) +r"""},
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_EdgeTransform(name, in_dim, out_dim, heads, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=18, depth=15):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Edge Transform,
        zlabel="""+ str(out_dim*heads) +r""",
        fill=\EdgeColor,
        opacity=0.7,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_lateral_connection(from_node, to_node, label="", pos=0.5):
    label_node = ""
    if label:
        label_node = r"""node[pos=""" + str(pos) + r""",above] {\small """ + label + r"""}"""
        
    return r"""\draw [connection] (""" + from_node +r"""-east)
-| node[pos=""" + str(pos) + """] {\midarrow} """ + label_node + r""" (""" + to_node +r"""-north);
"""

def to_lateral_connection_double_arrow(from_node, to_node, label="", pos=0.5, pos2=0.8):
    label_node = ""
    if label:
        label_node = r"""node[pos=""" + str(pos) + r""",above] {\small """ + label + r"""}"""
        
    return r"""\draw [connection] (""" + from_node +r"""-east)
-| node[pos=""" + str(pos) + """] {\midarrow} """ + label_node + r""" node[pos=""" + str(pos2) + """] {\midarrow} (""" + to_node +r"""-north);
"""

def to_residual_connection(from_node, to_node, pos=0.6):
    return r"""\draw [connection] (""" + from_node +r"""-east)
-| node[pos=""" + str(pos) + """] {\midarrow} (""" + to_node +r"""-south);
"""

def to_skip_input_connection(from_node, to_node, pos=0.6):
    return r"""\draw [connection] (""" + from_node +r"""-east)
|- node[pos=""" + str(pos) + """] {\midarrow} (""" + to_node +r"""-west);
"""

def to_AttentionCoeff(name, num_edges, heads, offset="(0,0,0)", to="(0,0,0)", width=2, height=12, depth=18):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Ball={
        name=""" + name +r""",
        caption=Attention,
        fill=\AttentionColor,
        opacity=0.8,
        radius="""+ str(width) +r""",
        logo=$\alpha$
        }
    };
"""

def to_Messages(name, nodes, heads, dim, offset="(0,0,0)", to="(0,0,0)", width=3, height=20, depth=25):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Messages,
        zlabel="""+ str(nodes) +r""" x """+ str(heads) +r""" x """+ str(dim) +r""",
        fill=\AttentionColor,
        opacity=0.6,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_HeadAverage(name, nodes, dim, offset="(0,0,0)", to="(0,0,0)", width=2.5, height=18, depth=20):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Heads Avg,
        zlabel="""+ str(nodes) +r""" x """+ str(dim) +r""",
        fill=\ConvColor,
        opacity=0.7,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_SkipConnection(name, nodes, dim, offset="(0,0,0)", to="(0,0,0)", width=2, height=15, depth=15):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Skip Connection,
        zlabel="""+ str(nodes) +r""" x """+ str(dim) +r""",
        fill=\PoolColor,
        opacity=0.6,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_ResidualAdd(name, nodes, dim, offset="(0,0,0)", to="(0,0,0)", width=2.5, height=20, depth=20):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Ball={
        name=""" + name +r""",
        caption=Residual Add,
        fill=\FcColor,
        opacity=0.8,
        radius="""+ str(width) +r""",
        logo=$+$
        }
    };
"""

# --- NEW: basic post-processing blocks (BatchNorm, GELU, Dropout) ---

def to_BatchNorm(name, offset="(0,0,0)", to="(0,0,0)", width=1.2, height=18, depth=18):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
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

def to_GELU(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=16, depth=16):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=GELU,
        fill=\FcReluColor,
        opacity=0.8,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_Dropout(name, offset="(0,0,0)", to="(0,0,0)", width=0.8, height=14, depth=14):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Dropout,
        fill=\PoolColor,
        opacity=0.4,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

# --- NEW: Linear block ---

def to_Linear(name, in_dim, out_dim, offset="(0,0,0)", to="(0,0,0)", width=2, height=18, depth=15):
    return r"""\pic[shift={"""+ offset +"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption=Linear,
        zlabel="""+ str(out_dim) +r""",
        fill=\FcColor,
        opacity=0.8,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

# Build the complete architecture following the table exactly
arch = [
    to_head('..'),
    to_cor(),
    to_extended_colors(),
    to_begin(),
]

print("Building GraphTransformer with Edge Features - STEP BY STEP DEBUG...")

# ===== STEP 0: x (node features) - (522 x 22) =====
print("Step 0: Node features (522 x 22)")
arch.append(
    to_Conv(
        "node_features",
        s_filer=NUM_FEATURES,
        n_filer=NUM_NODES,
        offset="(0,0,0)",
        to="(0,0,0)",
        height=45,
        width=6,
        depth=45,
        caption="Node Features"
    )
)

# ===== STEP 0': edge_index (connectivity) - (2 x 5) =====
print("Step 0': edge_index (2 x 5) - Topology governing propagation")
arch.append(
    to_EdgeIndex(
        "edge_index",
        NUM_EDGES,
        offset="(0,4,0)",
        to="(node_features-north)",
        width=1,
        height=8,
        depth=12
    )
)

# ===== STEP 1: Q/K/V layer 1 -> reshape (522 x 2 x 32) =====
print("Step 1: Q/K/V layer 1 reshape (522 x 2 x 32)")
prev_layer = "node_features"
qkv1_name = "qkv1"
arch.append(
    to_QKV_WithEdges(
        qkv1_name,
        NUM_FEATURES,
        D_HEAD_L1,
        NUM_HEADS_L1,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({prev_layer}-east)",
        width=2,
        height=35,
        depth=30
    )
)
arch.append(to_connection(prev_layer, qkv1_name))

# ===== STEP 2: α₁ coeficientes de atención (5 × 2) =====
print("Step 2: α₁ coeficientes de atención (5 × 2)")
attention1_name = "alpha1"
arch.append(
    to_AttentionCoeff(
        attention1_name,
        NUM_EDGES,
        NUM_HEADS_L1,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({qkv1_name}-east)",
        width=1.8,
        height=12,
        depth=18
    )
)
arch.append(to_connection(qkv1_name, attention1_name))

# LATERAL CONNECTION: edge_index provides topology
arch.append(to_lateral_connection_double_arrow("edge_index", attention1_name, "topology", 0.3, 0.7))

# ===== STEP 2': m₁ mensajes agregados (Σ α·V) - (522 × 2 × 64) =====
print("Step 2': m₁ mensajes agregados (Σ α·V) - (522 × 2 × 64)")
messages1_name = "messages1"
arch.append(
    to_Messages(
        messages1_name,
        NUM_NODES,
        NUM_HEADS_L1,
        D_HEAD_L1,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({attention1_name}-east)",
        width=3,
        height=25,
        depth=30
    )
)
arch.append(to_connection(attention1_name, messages1_name))

# ===== STEP 3: Promedio cabezas (concat=False) - (522 × 64) =====
print("Step 3: Promedio cabezas (concat=False) - (522 × 64)")
head_avg1_name = "head_avg1"
arch.append(
    to_HeadAverage(
        head_avg1_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({messages1_name}-east)",
        width=2.5,
        height=20,
        depth=25
    )
)
arch.append(to_connection(messages1_name, head_avg1_name))

# ===== STEP 3': r₁ skip connection (W₁·x) - (522 × 64) =====
print("Step 3': r₁ skip connection (W₁·x) - (522 × 64)")
skip1_name = "skip1"
arch.append(
    to_SkipConnection(
        skip1_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset="(0,-6,0)",
        to=f"({head_avg1_name}-south)",
        width=2,
        height=15,
        depth=20
    )
)
arch.append(to_skip_input_connection("node_features", skip1_name))

# ===== STEP 3'': h₁ = r₁ + m₁ (β desactivado) - (522 × 64) =====
print("Step 3'': h₁ = r₁ + m₁ (β desactivado) - (522 × 64)")
h1_name = "h1_output"
arch.append(
    to_ResidualAdd(
        h1_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset="(2,0,0)",
        to=f"({head_avg1_name}-east)",
        width=2,
        height=18,
        depth=20
    )
)
arch.append(to_connection(head_avg1_name, h1_name))
arch.append(to_residual_connection(skip1_name, h1_name))

# ===== STEP 4: BatchNorm 1 → GELU → Dropout =====
print("Step 4: BatchNorm 1 → GELU → Dropout")
arch.append(to_BatchNorm("BatchNorm_1", offset=f"({HORIZONTAL_SPACING},0,0)", to=f"({h1_name}-east)"))
arch.append(to_connection(h1_name, "BatchNorm_1"))
arch.append(to_GELU("GELU_1", offset="(2,0,0)", to=f"(BatchNorm_1-east)"))
arch.append(to_Dropout("Dropout_1", offset="(2,0,0)", to=f"(GELU_1-east)"))
arch.append(to_connection("BatchNorm_1", "GELU_1"))
arch.append(to_connection("GELU_1", "Dropout_1"))

# ===== STEP 5: Q/K/V layer 2 -> reshape (522 x 1 x 64) =====
print("Step 5: Q/K/V layer 2 reshape (522 x 1 x 64)")
prev_layer2 = "Dropout_1"
qkv2_name = "qkv2"
arch.append(
    to_QKV_WithEdges(
        qkv2_name,
        HIDDEN_DIM,
        D_HEAD_L2,
        NUM_HEADS_L2,
        offset="(4,0,0)",
        to=f"({prev_layer2}-east)",
        width=2.5,
        height=25,
        depth=30
    )
)
arch.append(to_connection(prev_layer2, qkv2_name))

# ===== STEP 6: α₂ coeficientes de atención (5 × 1) =====
print("Step 6: α₂ coeficientes de atención (5 × 1)")
attention2_name = "alpha2"
arch.append(
    to_AttentionCoeff(
        attention2_name,
        NUM_EDGES,
        NUM_HEADS_L2,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({qkv2_name}-east)",
        width=1.8,
        height=12,
        depth=18
    )
)
arch.append(to_connection(qkv2_name, attention2_name))

# LATERAL CONNECTION: edge_index provides topology
arch.append(to_lateral_connection_double_arrow("edge_index", attention2_name, "topology", 0.3, 0.7))

# ===== STEP 6': m₂ mensajes agregados (Σ α·V) - (522 × 1 × 64) =====
print("Step 6': m₂ mensajes agregados (Σ α·V) - (522 × 1 × 64)")
messages2_name = "messages2"
arch.append(
    to_Messages(
        messages2_name,
        NUM_NODES,
        NUM_HEADS_L2,
        D_HEAD_L2,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({attention2_name}-east)",
        width=3,
        height=25,
        depth=30
    )
)
arch.append(to_connection(attention2_name, messages2_name))

# ===== STEP 7: Promedio cabezas (concat=False) - (522 × 64) =====
print("Step 7: Promedio cabezas (concat=False) - (522 × 64)")
head_avg2_name = "head_avg2"
arch.append(
    to_HeadAverage(
        head_avg2_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({messages2_name}-east)",
        width=2.5,
        height=20,
        depth=25
    )
)
arch.append(to_connection(messages2_name, head_avg2_name))

# ===== STEP 7': r₂ skip connection (W₁·h₁) - (522 × 64) =====
print("Step 7': r₂ skip connection (W₁·h₁) - (522 × 64)")
skip2_name = "skip2"
arch.append(
    to_SkipConnection(
        skip2_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset="(0,-6,0)",
        to=f"({head_avg2_name}-south)",
        width=2,
        height=15,
        depth=20
    )
)
# Connect processed features to the second skip connection using the skip input style
arch.append(to_skip_input_connection("Dropout_1", skip2_name))

# ===== STEP 7'': h₂ = r₂ + m₂ (β desactivado) - (522 × 64) =====
print("Step 7'': h₂ = r₂ + m₂ (β desactivado) - (522 × 64)")
h2_name = "h2_output"
arch.append(
    to_ResidualAdd(
        h2_name,
        NUM_NODES,
        HIDDEN_DIM,
        offset="(2,0,0)",
        to=f"({head_avg2_name}-east)",
        width=2,
        height=18,
        depth=20
    )
)
arch.append(to_connection(head_avg2_name, h2_name))
arch.append(to_residual_connection(skip2_name, h2_name))

# ===== STEP 8: BatchNorm 2 =====
print("Step 8: BatchNorm 2")
arch.append(to_BatchNorm("BatchNorm_2", offset=f"({HORIZONTAL_SPACING},0,0)", to=f"({h2_name}-east)"))
arch.append(to_connection(h2_name, "BatchNorm_2"))

# ===== STEP 9: Linear 64 -> 32 =====
print("Step 9: Linear 64 -> 32")
linear1_name = "linear1"
arch.append(
    to_Linear(
        linear1_name,
        HIDDEN_DIM,
        HIDDEN_DIM//2,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"(BatchNorm_2-east)",
        width=2,
        height=18,
        depth=15
    )
)
arch.append(to_connection("BatchNorm_2", linear1_name))

# ===== STEP 10: Linear 32 -> 1 (ŷ) =====
print("Step 10: Linear 32 -> 1 (ŷ)")
final_linear = "linear_out"
arch.append(
    to_Linear(
        final_linear,
        HIDDEN_DIM//2,
        NUM_TARGETS,
        offset=f"({HORIZONTAL_SPACING},0,0)",
        to=f"({linear1_name}-east)",
        width=2,
        height=16,
        depth=10
    )
)
arch.append(to_connection(linear1_name, final_linear))

# SoftMax / Predictions
arch.append(to_SoftMax("predictions", NUM_TARGETS, offset="(2,0,0)", to=f"({final_linear}-east)", width=2, height=20, depth=8, caption="Pred"))
arch.append(to_connection(final_linear, "predictions"))

arch.append(to_end())

print("Step 4 complete - first layer post-processing added.")

def main():
    namefile = str(sys.argv[1]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main() 