# Incremental Learning (Online/Adaptive Learning) Implementation

## Overview
This implementation adds incremental learning capabilities to the face recognition system, allowing it to continuously improve and adapt by updating face embeddings when successful recognitions occur.

## Key Features

### 1. **Embedding Averaging**
- Maintains multiple embeddings per person (up to 20 by default)
- Adds new high-confidence embeddings to the database
- Compares against all stored embeddings for better accuracy

### 2. **Weighted Updating**
- Uses configurable alpha value (0.8 by default) for stability
- Prevents drift from outliers or poor lighting conditions
- Maintains embedding quality over time

### 3. **Hard Example Filtering**
- Filters out embeddings that are too far from existing ones
- Uses outlier threshold (0.3 by default) to reject bad embeddings
- Protects against incorrect recognitions affecting the database

### 4. **Adaptive Learning Controls**
- **Cooldown Period**: Minimum 5 seconds between updates for same person
- **Confidence Threshold**: Only learns from high-confidence recognitions (0.8+)
- **Update Frequency**: Updates every 5 successful recognitions
- **Memory Management**: Keeps only the most recent embeddings per person

## Configuration Parameters

All settings are configurable in `config/config.json`:

```json
"incremental_learning": {
  "enabled": true,                    // Enable/disable learning
  "update_cooldown": 5.0,            // Seconds between updates
  "min_confidence": 0.8,             // Minimum confidence for learning
  "update_frequency": 5,             // Update every N recognitions
  "max_embeddings_per_person": 20,   // Max embeddings to store
  "outlier_threshold": 0.3,          // Distance threshold for outliers
  "weighted_alpha": 0.8              // Weighting factor for stability
}
```

## How It Works

### Recognition Process
1. **Face Detection**: Detect face and extract embedding
2. **Anti-Spoofing**: Verify face is real (not photo/video)
3. **Recognition**: Compare with stored embeddings
4. **Learning Decision**: Check if conditions are met for learning
5. **Update Embeddings**: Add new embedding if learning is triggered
6. **Save Database**: Asynchronously save updated database

### Learning Triggers
Learning occurs when ALL conditions are met:
- ✅ Face passes anti-spoofing checks
- ✅ Recognition confidence > threshold (0.8)
- ✅ Cooldown period has passed (5+ seconds)
- ✅ Frequency counter reached (every 5 recognitions)
- ✅ New embedding is not an outlier

### Visual Indicators
- `[LEARNING]` appears in the label when system is updating
- `[LIVE:0.xx]` shows anti-spoofing confidence
- Green color for known live faces
- Yellow color for unknown live faces
- Red color for spoofed faces

## Benefits

### 1. **Improved Accuracy Over Time**
- System adapts to lighting changes
- Learns from different angles and expressions
- Accommodates gradual appearance changes

### 2. **Robust Against Errors**
- Outlier detection prevents bad embeddings
- Cooldown prevents over-updating
- Confidence thresholds ensure quality

### 3. **Automatic Adaptation**
- No manual retraining required
- Continuous improvement during normal use
- Maintains performance in changing conditions

## Monitoring and Logs

The system logs all learning activities:
- When embeddings are updated
- Why updates are rejected (outliers, low confidence, etc.)
- Database save operations
- Recognition statistics

Check `logs/live_recognition_YYYYMMDD.log` for detailed information.

## Technical Implementation

### Cosine Distance Calculation
```python
cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
distance = 1 - cos_sim
```

### Embedding Update Process
1. Extract new embedding from current frame
2. Validate against existing embeddings (outlier check)
3. Add to embedding list if valid
4. Trim to maximum embeddings if needed
5. Save updated database asynchronously

### Memory Management
- Stores multiple embeddings per person in memory
- Limits storage to prevent excessive memory usage
- Keeps most recent embeddings (FIFO queue)

## Performance Considerations

- **Asynchronous Saving**: Database saves don't block recognition
- **Memory Efficient**: Limited embedding storage per person
- **CPU Optimized**: Learning only triggers periodically
- **Storage Efficient**: JSON compression for embedding storage

This incremental learning system provides a self-improving face recognition experience that gets more accurate and robust over time while maintaining real-time performance.
