import sys
sys.path.append('../')
from pycore.tikzeng import *

# =============================================================
# GraphTransformer Architecture Diagram
# -------------------------------------------------------------
# This script creates a TikZ / LaTeX figure that visualises the
# architecture of the GraphTransformer model described in
# the user prompt.  Run it via
#   bash ../tikzmake.sh graph_transformer_arch
# (from inside the pyexamples folder) to obtain a PDF.
# =============================================================

# Editable hyper-parameters (match these to your model)
NUM_LAYERS   = 2      # total TransformerConv layers
HIDDEN_DIM   = 64     # hidden dimension (d_model)
NUM_HEADS    = 2      # attention heads
NUM_TARGETS  = 522    # output dimension (regression)

# Parametrize the input nodes and features
NUM_INPUT_NODES = 22      # number of input nodes (e.g., graph nodes)
NUM_INPUT_FEATURES = 522  # number of features per node

# Convenience helpers ---------------------------------------------------------

def to_Conv_suppressed(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" "):
    """Create a Conv box with suppressed automatic xlabel/zlabel like SoftMax does"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{" ","dummy"}},
        zlabel=" ",
        fill=\ConvColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Predictor(name, s_filer=1, n_filer=522, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=" "):
    """Create a Predictor box with custom s_filer and n_filer and SoftMax coloring"""
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity=0.8,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def transformer_conv(name, prev_name, shift=(3, 0, 0), caption=""):
    """Create a TransformerConv-like box plus connection from prev_name."""
    x_shift = f"({shift[0]},{shift[1]},{shift[2]})"
    # Add vertical spacing to caption - keep text together, then position entire block
    spaced_caption = f"{caption}\\\\[-5cm]~" if caption else ""
    box = to_Conv(
        name,
        s_filer=HIDDEN_DIM,
        n_filer=HIDDEN_DIM,
        offset=x_shift,
        to=f"({prev_name}-east)",
        height=20,  # smaller than input
        width=2,
        depth=20,   # smaller than input
        caption=spaced_caption,
    )
    conn = to_connection(prev_name, name)
    return box, conn

# Build the architecture list -------------------------------------------------
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
]

# Input layer ------------------------------------------------------------------
arch.append(
    to_Conv(
        "input",
        s_filer=NUM_INPUT_NODES,   # number of nodes
        n_filer=NUM_INPUT_FEATURES,      # number of features per node
        offset="(0,0,0)",
        to="(0,0,0)",
        height=60,  # bigger than TransformerConv
        width=5,
        depth=60,   # bigger than TransformerConv
        caption=f"Station Graph\\\\[-5cm]~",
    )
)

# TransformerConv layers ------------------------------------------------------
prev = "input"
for idx in range(1, NUM_LAYERS + 1):
    name = f"trconv{idx}"
    arch.append(
        to_Conv(
            name,
            s_filer=HIDDEN_DIM,
            n_filer=HIDDEN_DIM,
            offset="(2,0,0)",
            to=f"({prev}-east)",
            height=20,  # smaller than input
            width=5,
            depth=20,   # smaller than input
            caption=f"T.Conv {NUM_HEADS} head{'' if NUM_HEADS == 1 else 's'}\\\\[-25cm]~",
        )
    )
    arch.append(to_connection(prev, name))
    prev = name

# Prediction (MLP) layer -------------------------------------------------------
arch.append(
    to_Predictor(
        "pred",
        s_filer=1,
        n_filer=NUM_TARGETS,  # 522
        offset="(2,0,0)",
        to=f"({prev}-east)",
        width=2,
        height=60,
        depth=10,
        caption=f"Arrivals\\\\[-5cm]~",
    )
)
arch.append(to_connection(prev, "pred"))

# Finish ----------------------------------------------------------------------
arch.append(to_end())


def main():
    namefile = str(sys.argv[1]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main() 