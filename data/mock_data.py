def get_initial_data():
    """
    Returns a list of initial and new mock data chunks for the SANM.
    """
    initial_data = [
        {"0": 0.9, "1": 0.8, "2": 0.7},  # Cat
        {"3": 0.85, "1": 0.82, "2": 0.75, "4": 0.9},  # Dog
        {"5": 0.92, "1": 0.88, "2": 0.8},  # Lion
        {"0": 0.7, "6": 0.9}   # Similar Cat
    ]

    new_data = [
    # Similar to Existing
        {"0": 0.88, "1": 0.75, "2": 0.68, "7": 0.3},  # Slightly different cat
        {"3": 0.9, "1": 0.80, "2": 0.73, "4": 0.88, "8": 0.4},  # Slightly different dog
        {"5": 0.85, "1": 0.7, "2": 0.78, "9": 0.3}, # slightly different Lion

        # Distinct New Categories
        {"10": 0.9, "11": 0.7, "12": 0.6}, # Car
        {"10": 0.8, "11": 0.6, "13": 0.8}, # Truck
        {"14": 0.9, "15": 0.8, "16": 0.7}, # Apple
        {"17": 0.8, "18": 0.7, "19": 0.9}, # Pizza
        {"20": 0.9, "21": 0.8, "22": 0.8}, # Park
        {"23": 0.8, "24": 0.7, "25": 0.9}, # Home

        # Overlapping Concepts
        {"0": 0.8, "1": 0.7, "2": 0.6, "6": 0.8, "26": 0.9},   # Pet cat with similar attributes
        {"3": 0.7, "1": 0.6, "2": 0.65, "4": 0.8, "27": 0.8, "28": 0.8},  # Labrador (dog breed)

        # Noisy/Unrelated Data
         {"90": 0.2, "91": 0.4, "92": 0.1, "93": 0.3}   # Noise
    ]
    return initial_data + new_data

if __name__ == "__main__":
    # Example of how to use the function (optional)
    initial_data = get_initial_data()
    for data in initial_data:
        print(data)