import pandas as pd
import json
import os
from prepare_dataset import _row, to_multihot

def test_row():
    # Test single emotion index
    r1 = _row("test text", 2, 0.5, 0, 27, "US")
    print(f"Row 1 emotion: {r1['emotion_label']}")
    assert isinstance(r1['emotion_label'], str)
    assert json.loads(r1['emotion_label'])[27] == 1.0
    
    # Test list of indices
    r2 = _row("test text", 2, 0.5, 0, [2, 5, 27], "India")
    print(f"Row 2 emotion: {r2['emotion_label']}")
    vec = json.loads(r2['emotion_label'])
    assert vec[2] == 1.0
    assert vec[5] == 1.0
    assert vec[27] == 1.0
    assert vec[0] == 0.0

if __name__ == "__main__":
    test_row()
    print("Verification passed!")
