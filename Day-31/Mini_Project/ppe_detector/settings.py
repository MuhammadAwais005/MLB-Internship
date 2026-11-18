"""
Class names and drawing colours, shared by the training scripts and the app.

Roboflow exported the classes as '0', '1', '2', '3'. Their meaning was found by
looking at the labelled boxes (Day-31 Coding_Practice, class_examples.jpg).
"""

CLASS_NAMES = {0: "helmet", 1: "boots", 2: "person", 3: "vest"}

# BGR, the order OpenCV draws in
CLASS_COLORS = {
    0: (0, 165, 255),     # helmet - orange
    1: (180, 105, 255),   # boots  - pink
    2: (255, 160, 0),     # person - blue
    3: (0, 220, 120),     # vest   - green
}

# the equipment a worker is expected to wear
PPE_ITEMS = ("helmet", "vest", "boots")
