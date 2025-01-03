import networkx as nx
from data.mock_hkg import get_mock_hkg_data
import copy

class HKG_AG:
    def __init__(self):
        """
        Initializes the Hierarchical Knowledge Graph with Adaptive Granularity
        and loads data from mock_hkg.py
        """
        self.graph = nx.MultiDiGraph() # Use a MultiDiGraph to allow multiple edges between nodes
        self._load_mock_data()

    def _load_mock_data(self):
        """Loads mock data from data/mock_hkg.py."""
        hkg_data = get_mock_hkg_data()
        
        for node_data in hkg_data["nodes"]:
            self.add_node(
                node_id = node_data["id"],
                name = node_data["name"],
                layer = node_data["layer"],
                sanm_references = node_data["sanm_references"],
                data = node_data.get("data")
            )
        for edge_data in hkg_data["edges"]:
            self.add_edge(
                source_id = edge_data["source"],
                target_id = edge_data["target"],
                relation = edge_data["relation"],
                data = edge_data.get("data")
            )

    def add_node(self, node_id, name, layer, sanm_references, data=None):
        """
        Adds a new node to the knowledge graph.

        Args:
            node_id (str or int): Unique identifier for the node.
            name (str): Name or label of the node.
            layer (int): Layer in the hierarchy.
            sanm_references (list): List of references to SANM memory locations.
            data (dict, optional): Additional metadata for the node.
        """
        if self.graph.has_node(node_id):
            print(f"Warning: Node with id '{node_id}' already exists.")
            return
        
        node_data = {
            'name': name,
            'layer': layer,
            'sanm_references': sanm_references,
             'data': data if data else {},
        }
        
        self.graph.add_node(node_id, **node_data)
        print(f"Node '{name}' with id '{node_id}' added to layer {layer}.")

    def add_edge(self, source_id, target_id, relation, data = None):
        """
        Adds a new edge (relationship) between two nodes.

        Args:
            source_id (str or int): ID of the source node.
            target_id (str or int): ID of the target node.
             relation (str): Description of the relationship between nodes.
             data (dict, optional): Additional metadata for the edge
        """
        if not self.graph.has_node(source_id):
            print(f"Error: Source node with id '{source_id}' not found.")
            return
        if not self.graph.has_node(target_id):
            print(f"Error: Target node with id '{target_id}' not found.")
            return

        edge_data = {
            'relation': relation,
            'data': data if data else {}
        }
        self.graph.add_edge(source_id, target_id, **edge_data)
        print(f"Edge '{relation}' added from '{source_id}' to '{target_id}'.")
    
    def update_node_layer(self, node_id, new_layer, remap_edges = False):
        """Updates the layer of an existing node."""
        if not self.graph.has_node(node_id):
             print(f"Error: Node with id '{node_id}' not found.")
             return
        
        old_layer = self.graph.nodes[node_id]['layer']
        self.graph.nodes[node_id]['layer'] = new_layer
        print(f"Node '{node_id}' moved from layer: '{old_layer}' to layer: '{new_layer}'")

        if remap_edges:
            print(f"Remapping edges for node '{node_id}'")
            for source_id, target_id, key, data in self.graph.edges(keys=True, data=True):
                if source_id == node_id and self.graph.nodes[target_id]['layer'] < new_layer:
                    self.graph.remove_edge(source_id, target_id, key=key)
                    print(f"  Edge: {source_id} -> {target_id} removed.")
                elif target_id == node_id and self.graph.nodes[source_id]['layer'] < new_layer:
                    self.graph.remove_edge(source_id, target_id, key=key)
                    print(f"  Edge: {source_id} -> {target_id} removed.")
            print("Finished remapping edges")
        
    
    def update_node_data(self, node_id, data):
        """Updates data of an existing node"""
        if not self.graph.has_node(node_id):
             print(f"Error: Node with id '{node_id}' not found.")
             return
        self.graph.nodes[node_id]['data'] = data
        print(f"Node '{node_id}' data updated.")
    
    def update_edge_data(self, source_id, target_id, key, data):
        """Updates the data of an existing edge"""
        if not self.graph.has_edge(source_id, target_id, key = key):
            print(f"Error: Edge from '{source_id}' to '{target_id}' with key '{key}' not found.")
            return
        self.graph.edges[source_id, target_id, key]['data'] = data
        print(f"Edge from '{source_id}' to '{target_id}' data updated.")
    
    def get_node(self, node_id):
        """Returns the node information from the graph"""
        if not self.graph.has_node(node_id):
            print(f"Error: Node with id '{node_id}' not found.")
            return None
        return self.graph.nodes[node_id]
    
    def get_edge(self, source_id, target_id, key):
        """Returns the edge information from the graph"""
        if not self.graph.has_edge(source_id, target_id, key = key):
            print(f"Error: Edge from '{source_id}' to '{target_id}' with key '{key}' not found.")
            return None
        return self.graph.edges[source_id, target_id, key]
    
    def get_nodes_in_layer(self, layer):
         """Returns all nodes in a specific layer."""
         nodes_in_layer = [
             (node_id, data) for node_id, data in self.graph.nodes(data=True) if data['layer'] == layer
             ]
         return nodes_in_layer

    def display_graph(self):
        """Displays the knowledge graph information (nodes and edges)."""
        print("\n--- Knowledge Graph ---")
        print("Nodes:")
        for node_id, data in self.graph.nodes(data=True):
            print(f"  Node ID: {node_id}, Name: {data['name']}, Layer: {data['layer']}, SANM Refs: {data['sanm_references']}, Data: {data['data']}")
        print("\nEdges:")
        for source_id, target_id, key, data in self.graph.edges(keys = True, data = True):
            print(f"  Edge: {source_id} -> {target_id} (Relation: {data['relation']}, Data: {data['data']}, Key: {key})")
        print("----------------------")
    
    def merge_nodes(self, node_ids, new_node_id, new_node_name, new_layer, new_node_data = None):
      """Merges multiple nodes into a single abstract node."""

      if not all(self.graph.has_node(node_id) for node_id in node_ids):
            print("Error: One or more of the specified nodes do not exist.")
            return
      
      if not all(isinstance(self.graph.nodes[node_id]['layer'], int) for node_id in node_ids):
          print("Error: One or more nodes are not on a specific layer.")
          return
      
      if not all(self.graph.nodes[node_id]['layer'] < new_layer for node_id in node_ids):
           print(f"Error: The provided nodes are not all on a lower layer than: '{new_layer}'")
           return
      
      #Create the new node data
      merged_sanm_references = []
      for node_id in node_ids:
           merged_sanm_references.extend(self.graph.nodes[node_id]['sanm_references'])
      merged_data = new_node_data if new_node_data else {}
      
      self.add_node(node_id=new_node_id, name = new_node_name, layer = new_layer, sanm_references = merged_sanm_references, data=merged_data)
      
      #Remap the edges
      edges_to_remap = list(self.graph.edges(keys=True, data=True))
      for source_id, target_id, key, data in edges_to_remap:
           if source_id in node_ids:
               self.add_edge(source_id = new_node_id, target_id = target_id, relation = data["relation"], data = data.get("data"))
               self.graph.remove_edge(source_id, target_id, key = key)
               print(f"Remapped Edge: '{source_id}' -> '{target_id}' to '{new_node_id}' -> '{target_id}'")
           elif target_id in node_ids:
                self.add_edge(source_id = source_id, target_id = new_node_id, relation = data["relation"], data = data.get("data"))
                self.graph.remove_edge(source_id, target_id, key=key)
                print(f"Remapped Edge: '{source_id}' -> '{target_id}' to '{source_id}' -> '{new_node_id}'")
      
      #Remove the old nodes
      for node_id in node_ids:
         self.graph.remove_node(node_id)
         print(f"Removed Node: '{node_id}'")
      
      print("Node Merge Completed")

if __name__ == "__main__":
    hkg = HKG_AG()
    hkg.display_graph()
    
    #get the node information
    print(f"\nNode information: {hkg.get_node(node_id = 'cat')}")
    print(f"\nEdge information: {hkg.get_edge(source_id = 'cat', target_id = 'animal', key = 0)}")
    
    #get nodes in a layer
    print(f"\nNodes in Layer 0: {hkg.get_nodes_in_layer(layer = 0)}")

    #update the layer of the cat, remap edges
    hkg.update_node_layer(node_id = "cat", new_layer = 2, remap_edges = True)
    hkg.display_graph()
    
    #get the node information
    print(f"\nNode information: {hkg.get_node(node_id = 'cat')}")
    print(f"\nEdge information: {hkg.get_edge(source_id = 'cat', target_id = 'animal', key = 0)}")
    
    #get nodes in a layer
    print(f"\nNodes in Layer 0: {hkg.get_nodes_in_layer(layer = 0)}")
    
    #merge nodes
    hkg.merge_nodes(node_ids = ["cat", "dog", "lion"], new_node_id = "animals", new_node_name = "Animals", new_layer = 3)
    hkg.display_graph()
    
    #get the node information
    print(f"\nNode information: {hkg.get_node(node_id = 'animals')}")
    
    #get nodes in a layer
    print(f"\nNodes in Layer 1: {hkg.get_nodes_in_layer(layer = 1)}")