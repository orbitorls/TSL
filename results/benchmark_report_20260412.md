# TSL-51 Thai Sign Language - Benchmark Report

## Model Information
- **Model**: tsl51_gru_20260412_220010.pt
- **Architecture**: Bidirectional GRU (2 layers)
- **Hidden Dim**: 128
- **Parameters**: 534,323 (~0.53M)
- **Input Dim**: 162 features
- **Classes**: 51 Thai signs

## Benchmark Setup
- **Test Samples**: 5,494
- **Split**: 80% train / 20% test
- **Stratified**: Yes

## Performance Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 98.80% |
| **Precision** | 98.85% |
| **Recall** | 98.80% |
| **F1-Score** | 98.81% |

## Inference Speed
- **Total Time**: 2.23 seconds
- **Speed**: 2,460 inferences/sec

## Top 10 Misclassifications
| True -> Predicted | Count |
|-------------------|-------|
| sorry -> marry | 6 |
| marry -> school | 5 |
| you -> grandmother | 4 |
| I -> like | 4 |
| I -> lonely | 4 |
| ask -> grandmother | 3 |
| girlfriend -> what | 3 |
| like -> I | 3 |
| older sibling -> sleepy | 3 |
| travel -> grandmother | 3 |

## Per-Class Accuracy Summary
- **100% accuracy**: 36 classes
- **80-99%**: 14 classes
- **<80%**: 1 class

## Conclusion
The model achieves **98.80% accuracy** on the test set, demonstrating excellent performance for Thai Sign Language recognition. The inference speed of ~2,460 samples/second makes it suitable for real-time applications.