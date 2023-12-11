# Code adapted from https://github.com/mathematiger/ExplwCE/blob/master/visualization.py
import colorsys
import logging
import math
import os
import re

import dgl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matplotlib")
logger.setLevel(logging.WARNING)


def generate_colors(num_colors):
    # Define the number of distinct hues to use
    num_hues = num_colors + 1
    # Generate a list of evenly spaced hues
    hues = [i / num_hues for i in range(num_hues)]
    # Shuffle the hues randomly
    # random.shuffle(hues)
    saturations = []
    # saturations = [0.8 for _ in range(num_colors)]
    values = []
    # values = [0.4 for _ in range(num_colors)]
    for i in range(num_colors):
        if i % 2 == 0:
            values.append(0.4)
            saturations.append(0.4)
        else:
            values.append(0.8)
            saturations.append(0.7)
    # Convert the hues, saturations, and values to RGB colors
    colors = [
        colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, saturations, values)
    ]
    # Convert the RGB colors to hexadecimal strings
    hex_colors = [
        f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors
    ]
    return hex_colors


def adjust_caption_size_exp(caption_length, max_size=18, min_size=8, rate=0.1):
    font_size = max_size * math.exp(-rate * caption_length)
    return max(min_size, min(font_size, max_size))


def visualize_hd(
    hd_graph,
    inverse_indices,
    file_name,
    target_dir=None,
    caption=None,
    with_labels=True,
):
    try:
        plt.clf()
    except Exception as e:
        print(f"An exception occurred while clearing the plot: {e}")

    list_all_nodetypes = [
        ntype for ntype in hd_graph.ntypes if len(hd_graph.nodes(ntype)) > 0
    ]
    # create data for legend and caption
    curent_nodetypes_to_all_nodetypes = [
        [count, hd_graph.ntypes.index(_)] for count, _ in enumerate(list_all_nodetypes)
    ]

    number_of_node_types_for_colors = len(curent_nodetypes_to_all_nodetypes)
    colors = generate_colors(number_of_node_types_for_colors)

    # create nx graph to visualize
    Gnew = nx.Graph()
    homdata = dgl.to_homogeneous(hd_graph)
    homdata.edge_index = torch.stack(homdata.edges(), dim=0)
    Gnew.add_nodes_from(list(range(homdata.num_nodes())))

    # add edges
    list_edges_start, list_edges_end = homdata.edge_index.tolist()
    nodes_to_explain = []
    item_count = 0
    for item, tensors in inverse_indices.items():
        target_node_ids = tensors.tolist()
        if tensors.numel() > 0:
            for count, nitem in enumerate(homdata.ndata["_TYPE"]):
                if nitem.item() == hd_graph.ntypes.index(item):
                    item_count += 1
                    if item_count in target_node_ids:
                        nodes_to_explain.append(count + 1)

    label_dict = {}
    node_color = []
    ntypes_list = homdata.ndata["_TYPE"].tolist()
    for count, item in enumerate(ntypes_list):
        node_label_to_index = list_all_nodetypes.index(hd_graph.ntypes[item])
        label_dict[count] = list_all_nodetypes[node_label_to_index][:3]
        if count in nodes_to_explain:
            node_color.append("#6E4B4B")
        else:
            node_color.append(colors[node_label_to_index])

    list_edges_for_networkx = list(zip(list_edges_start, list_edges_end))
    Gnew.add_edges_from(list_edges_for_networkx)
    # plt
    options = {"with_labels": "True", "node_size": 500}
    nx.draw(Gnew, node_color=node_color, **options, labels=label_dict)
    # create legend
    patch_list = []
    name_list = []
    for i in range(len(list_all_nodetypes)):
        cc = curent_nodetypes_to_all_nodetypes[i][1]
        patch_list.append(plt.Circle((0, 0), 0.1, fc=colors[i]))
        name_list.append(list_all_nodetypes[i])

    # create caption
    special_node_color = "#6E4B4B"
    special_node_label = "Target Node"
    patch_list.append(Patch(color=special_node_color))
    name_list.append(special_node_label)
    name_list = [name[1:] if name[0] == "_" else name for name in name_list]
    if caption:
        caption_text = caption
        caption_size = adjust_caption_size_exp(
            caption_length=len(caption), max_size=18, min_size=8, rate=0.1
        )
        caption_position = (0.5, -0.1)

    # folder to save in
    if target_dir:
        name_plot_save = f"{target_dir}/{file_name}"
    else:
        name_plot_save = f"results/exp_visualizations/{file_name}"
    directory = os.path.dirname(name_plot_save)
    os.makedirs(directory, exist_ok=True)

    if with_labels:
        # Define the file paths
        file_path_with_legend = f"{name_plot_save}.png"
        # Create legend and caption separately to avoid overlapping
        plt.legend(patch_list, name_list, loc="lower left")
        if caption:
            # Save the figure with legend and caption
            plt.figtext(*caption_position, caption_text, ha="center", size=caption_size)
        plt.savefig(file_path_with_legend, bbox_inches="tight")

        # Show the plot
        plt.show()
    else:
        # Define the file paths
        file_path_wo_legend = f"{name_plot_save}_wo.png"
        # Save the figure without legend and caption
        plt.savefig(file_path_wo_legend, bbox_inches="tight")

        # Show the plot
        plt.show()
