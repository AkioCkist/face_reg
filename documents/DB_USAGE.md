# Face Database Generator

This file creates a more accurate face recognition database by:

1. **Processing multiple face images** - Uses all available person*.jpg files in the directory
2. **Multiple embeddings per person** - Stores multiple embeddings for each person to improve recognition accuracy
3. **Data augmentation** - Creates slightly rotated and brightness-adjusted versions to improve robustness
4. **Adaptive face detection** - Falls back to alternative detection methods if the primary one fails
5. **Configuration-based settings** - Uses settings from config.json where available

## How to Use

1. Place your face images in the directory as:
   - person1.jpg
   - person2.jpg
   - person3.jpg
   - etc.

2. Run the face_db.py script:
   ```
   python face_db.py
   ```

3. When prompted, you can:
   - Use automatic naming (Person 1, Person 2, etc.)
   - Provide custom names for each person

4. The script will create a face_db.json file with the embeddings

## Technical Improvements

- **Multiple detection backends**: Uses RetinaFace by default, falls back to others (mediapipe, mtcnn, opencv)
- **Data augmentation**: Creates variations of each face with small rotations (±5°) and brightness adjustments (±10%)
- **Multiple embeddings per person**: Stores all embeddings to improve matching accuracy
- **Error handling**: Gracefully handles detection failures and provides informative logs
- **Progress tracking**: Uses tqdm to show progress bars during processing

## Output Format

The generated face_db.json file has a structure like:

```json
{
  "Person Name": {
    "embeddings": [
      [...embedding vector 1...],
      [...embedding vector 2...],
      ...
    ]
  },
  ...
}
```

This improved format allows the recognition system to match against multiple reference embeddings per person, significantly improving accuracy and robustness.
