def get_mock_hkg_data():
    """
    Returns a more complex mock knowledge graph data structure.
    """
    hkg_data = {
        "nodes": [
             {"id": "animal", "name": "Animal", "layer": 2, "sanm_references": [0, 1, 2], "data": {"description": "Living organisms that feed on organic matter."}},
            {"id": "cat", "name": "Cat", "layer": 1, "sanm_references": [0], "data": {"lifespan": "12-15 years"}},
            {"id": "dog", "name": "Dog", "layer": 1, "sanm_references": [1], "data": {"lifespan": "10-13 years"}},
            {"id": "lion", "name": "Lion", "layer": 1, "sanm_references": [2], "data": {"habitat": "Africa"}},
             {"id": "domestic", "name": "Domestic", "layer": 2, "sanm_references": [], "data": {"description": "Animals that have been tamed or kept as a pet"}},
            {"id": "pet", "name": "Pet", "layer": 2, "sanm_references":[], "data": {"description": "An animal kept for companionship."}},
            {"id": "labrador", "name": "Labrador Retriever", "layer": 0, "sanm_references":[1], "data": {"origin": "Newfoundland"}},
            {"id": "mammal", "name": "Mammal", "layer": 3, "sanm_references": [], "data":{"description": "Warm-blooded vertebrate animals"}},
             {"id": "vehicle", "name":"Vehicle", "layer": 2, "sanm_references": [], "data": {"description": "A thing used for transporting people or goods."}},
              {"id": "transportation", "name":"Transportation", "layer": 3, "sanm_references":[], "data": {"description": "The act or process of transporting or being transported."}},
            {"id": "car", "name": "Car", "layer": 1, "sanm_references":[], "data": {"type": "sedan"}},
            {"id": "truck", "name": "Truck", "layer": 1, "sanm_references":[], "data": {"type": "pickup"}},
            {"id": "food", "name": "Food", "layer": 2, "sanm_references": [], "data": {"description": "Any nutritious substance that people or animals eat or drink"}},
            {"id": "apple", "name": "Apple", "layer": 1, "sanm_references":[], "data": {"type": "fruit"}},
            {"id": "pizza", "name": "Pizza", "layer": 1, "sanm_references":[], "data": {"type": "meal"}},
             {"id": "place", "name":"Place", "layer": 2, "sanm_references": [], "data": {"description": "A particular position or point in space."}},
            {"id": "park", "name":"Park", "layer": 1, "sanm_references":[], "data": {"type":"outdoor"}},
            {"id": "home", "name":"Home", "layer":1, "sanm_references":[], "data": {"type":"indoor"}}
        ],
        "edges": [
            {"source": "cat", "target": "animal", "relation": "is_a", "data": {"weight": "2 - 10 kg"}},
             {"source": "dog", "target": "animal", "relation": "is_a", "data": {"weight": "5 - 40 kg"}},
            {"source": "lion", "target": "animal", "relation": "is_a", "data": {"weight": "150 - 250 kg"}},
            {"source": "cat", "target": "pet", "relation": "is_a", "data":{}},
            {"source": "dog", "target": "pet", "relation": "is_a", "data": {}},
            {"source": "labrador", "target": "dog", "relation": "is_a", "data":{}},
            {"source": "animal", "target": "mammal", "relation":"is_a", "data":{}},
             {"source":"cat", "target":"domestic", "relation":"has_attribute", "data": {}},
             {"source":"dog", "target":"domestic", "relation":"has_attribute", "data":{}},
            {"source": "vehicle", "target":"transportation", "relation":"is_a", "data":{}},
             {"source": "car", "target":"vehicle", "relation":"is_a", "data":{}},
            {"source":"truck", "target":"vehicle", "relation":"is_a", "data":{}},
            {"source": "apple", "target": "food", "relation": "is_a", "data":{}},
            {"source": "pizza", "target": "food", "relation":"is_a", "data":{}},
            {"source":"park", "target": "place", "relation":"is_a", "data": {}},
            {"source": "home", "target":"place", "relation":"is_a", "data":{}}
        ]
    }
    return hkg_data

if __name__ == "__main__":
    hkg_data = get_mock_hkg_data()
    print("--- Mock Knowledge Graph Data ---")
    print("Nodes:")
    for node in hkg_data["nodes"]:
        print(f"  {node}")
    print("\nEdges:")
    for edge in hkg_data["edges"]:
        print(f"  {edge}")