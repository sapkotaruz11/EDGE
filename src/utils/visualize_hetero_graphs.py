# Code adapted from https://github.com/mathematiger/ExplwCE/blob/master/visualization.py
import colorsys
import logging
import math
import os
import re

import dgl
import matplotlib.patches as patches
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import networkx as nx
import torch

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
    node_id=None,
    file_name=None,
    target_dir=None,
    caption=None,
    with_labels=True,
    edge_label_flag=False,
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
    try:
        nodes_to_explain = [
            item
            for sublist in hd_graph.ndata[dgl.NID].values()
            for item in sublist.tolist()
        ].index(node_id)
    except:
        nodes_to_explain = -1
        edge_label_flag = False

    label_dict = {}
    node_color = []
    ntypes_list = homdata.ndata["_TYPE"].tolist()
    for count, item in enumerate(ntypes_list):
        node_label_to_index = list_all_nodetypes.index(hd_graph.ntypes[item])
        label_dict[count] = list_all_nodetypes[node_label_to_index][:3]
        if count == nodes_to_explain:
            node_color.append("#6E4B4B")
        else:
            node_color.append(colors[node_label_to_index])
    edge_types = [hd_graph.etypes[item] for item in homdata.edata["_TYPE"].tolist()]
    list_edges_for_networkx = list(zip(list_edges_start, list_edges_end))
    edge_labels = {
        edge: etype
        for edge, etype in zip(list_edges_for_networkx, edge_types)
        if edge[0] == nodes_to_explain
    }
    unique_edge_types = set(edge_labels.values())

    # Generate a color palette
    color_palette = plt.cm.get_cmap(
        "hsv", len(unique_edge_types)
    )  # Using HSV colormap for variety
    edge_type_color = {
        etype: color_palette(i) for i, etype in enumerate(unique_edge_types)
    }

    # Apply colors to edges based on type
    edge_colors = [edge_type_color[etype] for etype in unique_edge_types]

    # Create a list of colors for each edge in the graph
    Gnew.add_edges_from(list_edges_for_networkx)
    # plt
    options = {"with_labels": "True", "node_size": 500}
    if edge_label_flag:
        nx.draw(
            Gnew,
            node_color=node_color,
            edge_color=edge_colors,
            **options,
            labels=label_dict,
        )
    else:
        nx.draw(
            Gnew,
            node_color=node_color,
            **options,
            labels=label_dict,
        )
    # create legend
    patch_list = []
    name_list = []
    for i in range(len(list_all_nodetypes)):
        cc = curent_nodetypes_to_all_nodetypes[i][1]
        patch_list.append(plt.Circle((0, 0), 0.1, fc=colors[i]))
        name_list.append(list_all_nodetypes[i])

    # create target node label
    if node_id is not None:
        special_node_color = "#6E4B4B"
        special_node_label = "Target Node"
        patch_list.append(Patch(color=special_node_color))
        name_list.append(special_node_label)
    name_list = [name[1:] if name[0] == "_" else name for name in name_list]

    # create caption
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
        # Create a legend for edge colors
        # Create and place the legend for node colors

        node_legend = plt.legend(
            patch_list,
            name_list,
            title="Node Types",
            loc="upper left",
            bbox_to_anchor=(-0.01, 0),  # Adjust these values
            borderaxespad=0.0,
        )
        if edge_label_flag:
            # Create and place the legend for edge colors
            edge_patch_list = [
                plt.Line2D([0], [0], color=color, label=etype, linewidth=2)
                for etype, color in edge_type_color.items()
            ]
            edge_legend = plt.legend(
                handles=edge_patch_list,
                title="Edge Types",
                loc="upper right",
                bbox_to_anchor=(1.05, 0),
                borderaxespad=0.0,
            )
            # Add the node legend back to the plot
            plt.gca().add_artist(node_legend)

        # Define the file paths
        if node_id > 0:
            file_path_with_legend = f"{name_plot_save}_{node_id}.png"
        else:
            file_path_with_legend = f"{name_plot_save}.png"
        if caption:
            # Save the figure with legend and caption
            plt.figtext(*caption_position, caption_text, ha="center", size=caption_size)
        plt.savefig(file_path_with_legend, bbox_inches="tight")
        plt.tight_layout()
        # Show the plot
        plt.show()
    else:
        # Define the file paths
        file_path_wo_legend = f"{name_plot_save}_wo.png"
        # Save the figure without legend and caption
        plt.savefig(file_path_wo_legend, bbox_inches="tight")

        # Show the plot
        plt.show()
