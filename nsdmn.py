from sanm import SANM
from hkg_ag import HKG_AG
from csam import CSAM
from data.mock_data import get_initial_data

class NSDMN:
    def __init__(self, feature_dimension):
        """
        Initializes the Neuro-Symbolic Dynamic Memory Network.
        """
        self.sanm = SANM(feature_dimension=feature_dimension, similarity_threshold=0.6, num_trees=10)
        self.hkg = HKG_AG()
        self.csam = CSAM(feature_dimension = feature_dimension)
        self._load_initial_data()
        print("NSDMN initialized.")

    def _load_initial_data(self):
          """Loads the initial data for SANM"""
          initial_data = get_initial_data()
          print("\nLoading initial data:")
          for data_chunk in initial_data:
              self.sanm.add(data_chunk)
          self.sanm.save_index()
    
    def query(self, query_chunk):
         """
        Processes a query using SANM, HKG-AG, and CSAM.

        Args:
            query_chunk (dict): A dictionary representing the sparse
                                       representation of the query.
        Returns:
            dict: A dictionary mapping node_id to the attention score.
         """
         print("\n--- Processing Query ---")
         #Get the query results from SANM
         sanm_results = self.sanm.query(query_chunk)

         #Get all of the nodes from HKG_AG
         hkg_nodes = [(node_id, data) for node_id, data in self.hkg.graph.nodes(data=True)]

         #Get the attention scores from CSAM
         attention_scores = self.csam.attend(query_chunk=query_chunk, hkg_nodes = hkg_nodes)
         print("--- Query Processing Completed ---")
         return attention_scores


if __name__ == "__main__":
    # Determine the feature dimension (max feature index + 1)
    initial_data = get_initial_data()
    feature_dimension = 0
    for data_chunk in initial_data:
      for feature in data_chunk:
          feature_dimension = max(feature_dimension, int(feature))
    feature_dimension += 1

    # Initialize the NSDMN
    nsdmn = NSDMN(feature_dimension = feature_dimension)

    # Create a mock query
    mock_query = {"1": 0.9, "3": 0.7}
    #process the query
    attention_scores = nsdmn.query(mock_query)
    
    #print out the results
    print("\nFinal Attention Scores:")
    for node_id, score in attention_scores.items():
         print(f"  Node: {node_id}, Score: {score:.2f}")

    # Create a mock query
    mock_query_2 = {"2": 0.8, "5": 0.95}
    #process the query
    attention_scores_2 = nsdmn.query(mock_query_2)
    
    #print out the results
    print("\nFinal Attention Scores:")
    for node_id, score in attention_scores_2.items():
         print(f"  Node: {node_id}, Score: {score:.2f}")