"""Video inference with sliding window and temporal smoothing."""



class VideoInference:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict_with_sliding_window(self, frames, window_size=30, stride=15):
        """Process video with sliding window."""
        windows = self._extract_windows(frames, window_size, stride)
        predictions, confidences = [], []
        for window in windows:
            features = self.predictor.extract_features(window)
            pred, conf = self.predictor.predict(features, return_top_k=1)
            predictions.append(pred)
            confidences.append(conf)
        return self._confidence_weighted_vote(predictions, confidences)

    def _extract_windows(self, frames, size, stride):
        """Extract sliding windows from frames."""
        return [frames[i:i+size] for i in range(0, len(frames)-size+1, stride)]

    def _confidence_weighted_vote(self, preds, confs):
        """Weighted voting by confidence."""
        if not preds:
            return None, 0
        weighted = {}
        for p, c in zip(preds, confs, strict=False):
            weighted[p] = weighted.get(p, 0) + c
        return max(weighted, key=weighted.get), max(confs)
