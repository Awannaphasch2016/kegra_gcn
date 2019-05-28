import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# For color mapping
import matplotlib.colors as colors
import matplotlib.cm as cmx


G = nx.Graph()
G.add_node("kind1")
G.add_node("kind2")
G.add_node("Obsolete")
G.add_node("Draft")
G.add_node("Release")
G.add_node("Initialisation")

# You were missing the position.
pos = nx.spring_layout(G)
val_map = {'kind1': 2, 'kind2': 2, 'Obsolete': 2, 'Initialisation': 1, 'Draft': 4, 'Release': 3}
values = [val_map.get(node, 0) for node in G.nodes()]

# Color mapping
jet = cm = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin=0, vmax=max(values))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# Using a figure to use it as a parameter when calling nx.draw_networkx
f = plt.figure(1)
ax = f.add_subplot(1, 1, 1)

# for label in val_map:
    # ax.plot([0], [0],
    #         color=scalarMap.to_rgba(val_map[label]),
    #         label=label)

for value in set(values):
    ax.plot([0], [0],
            color=scalarMap.to_rgba(value),
            label=value)

# Just fixed the color map
nx.draw_networkx(G, pos, cmap=jet, vmin=0, vmax=max(values),
                 node_color=values,
                 with_labels=False, ax=ax)

# Here is were I get an error with your code
# nodes = nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)

# Setting it to how it was looking before.
plt.axis('off')
f.set_facecolor('w')

plt.legend(loc='center')

f.tight_layout()
plt.show()