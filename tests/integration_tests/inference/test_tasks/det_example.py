import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    BaseAiCOCOImageInputDataModel,
    BaseAiCOCOImageOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiImage
from flavor.serve.inference.inference_models import BaseAiCOCOImageInferenceModel
from flavor.serve.inference.strategies import AiCOCODetectionOutputStrategy


class DetectionInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(self):
        self.formatter = AiCOCODetectionOutputStrategy()
        super().__init__()

    def define_inference_network(self) -> Callable:
        ckpt_path = os.path.join(os.getcwd(), "best.pt")
        if not os.path.exists(ckpt_path):
            from urllib.request import urlretrieve

            urlretrieve(
                "https://github.com/sevdaimany/YOLOv8-Medical-Imaging/raw/master/runs/detect/train/weights/best.pt",
                ckpt_path,
            )
        return YOLO(ckpt_path)

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [
            {"name": "RBC", "display": True},
            {"name": "WBC", "display": True},
            {"name": "Platelets", "display": True},
        ]
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        image = cv2.imread(files[0])
        image = image.astype(np.float32)

        return image, None, None

    def inference(self, x: np.ndarray) -> Any:
        return self.network.predict(x, conf=0.7)[0]

    def postprocess(self, model_out: Any, **kwargs) -> Dict[str, Any]:

        format_output = {
            "bbox_pred": [],
            "cls_pred": [],
            "confidence_score": [],
        }

        for obj in model_out.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = obj
            format_output["bbox_pred"].append([int(x1), int(y1), int(x2), int(y2)])
            cls_pred = np.zeros(3)
            cls_pred[int(class_id)] = 1
            format_output["cls_pred"].append(cls_pred)
            format_output["confidence_score"].append(score)

        return format_output

    def output_formatter(
        self,
        model_out: Dict[str, Any],
        images: Sequence[AiImage],
        categories: Sequence[Dict[str, Any]],
        regressions: Sequence[Dict[str, Any]],
        **kwargs
    ) -> BaseAiCOCOImageOutputDataModel:
        output = self.formatter(
            model_out=model_out, images=images, categories=categories, regressions=regressions
        )
        return output


app = InferAPP(
    infer_function=DetectionInferenceModel(),
    input_data_model=BaseAiCOCOImageInputDataModel,
    output_data_model=BaseAiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
