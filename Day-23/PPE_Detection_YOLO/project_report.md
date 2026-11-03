# Mini Project: Object Detection Using YOLO11

## 1. Objective

To perform object detection using a pretrained YOLO11n model on a public PPE detection dataset.

## 2. Dataset

- Source: Roboflow Universe
- Total images: 1,159
- Training images: 738
- Validation images: 315
- Testing images: 106
- Number of classes: 7

## 3. Model

- Model: YOLO11n
- Weights: yolo11n.pt
- Confidence threshold: 0.25
- Training: No custom training performed
- Task: Object detection inference

## 4. Observations

- The model successfully detected persons.
- The model did not detect safety helmets as a separate PPE class.
- A window was incorrectly classified as a TV.
- The incorrect TV prediction had 33% confidence.
- The TV prediction was retained because its confidence exceeded 0.25.

## 5. Analysis

The pretrained model demonstrated its ability to detect
general objects such as persons. However, it struggled
with PPE-specific categories because it was not trained
on our custom PPE dataset.

The incorrect TV prediction demonstrates that object
detection models can produce false positives.

## 6. Conclusion

The experiment demonstrated how pretrained YOLO11
performs inference on a custom object detection dataset.

Fine-tuning on the PPE dataset would be the next step
towards building a PPE-specific detection system.
