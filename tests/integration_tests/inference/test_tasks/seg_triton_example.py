import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    BaseAiCOCOImageInputDataModel,
    BaseAiCOCOImageOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiImage
from flavor.serve.inference.inference_models import (
    BaseAiCOCOImageInferenceModel,
    TritonInferenceModel,
    TritonInferenceModelSharedSystemMemory,
)
from flavor.serve.inference.strategies import AiCOCOSegmentationOutputStrategy


class SegmentationTritonInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(
        self,
        triton_url: str = "triton:8000",
        model_name: str = "toyseg",
        model_version: str = "",
        is_shared_memory: bool = False,
    ):
        self.formatter = AiCOCOSegmentationOutputStrategy()

        self.triton_url = triton_url
        self.model_name = model_name
        self.model_version = model_version
        self.is_shared_memory = is_shared_memory
        super().__init__()

    def define_inference_network(self) -> Callable:
        if self.is_shared_memory:
            return TritonInferenceModelSharedSystemMemory(
                self.triton_url, self.model_name, self.model_version
            )
        else:
            return TritonInferenceModel(self.triton_url, self.model_name, self.model_version)

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [
            {"name": "Background", "display": False},
            {"name": "Foreground 1", "display": True},
            {"name": "Foreground 2", "display": True},
        ]
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        img = cv2.imread(files[0])
        return img, None, None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)
        data = np.transpose(data, (2, 0, 1))  # h, w, c -> c, h, w
        data = np.expand_dims(data, axis=0)  # c, h, w -> 1, c, h, w
        return data

    def inference(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        return self.network.forward({"input": x})

    def postprocess(
        self, out_dict: Dict[str, np.ndarray], metadata: Optional[Any] = None
    ) -> np.ndarray:
        out = out_dict["logits"][0]  # 1, c, h, w -> c, h, w
        onehot_out = np.zeros_like(out, dtype=np.int8)
        out = np.argmax(out, axis=0)
        for i in range(len(onehot_out)):
            onehot_out[i] = out == i
        return onehot_out

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: Sequence[Dict[str, Any]],
        **kwargs
    ) -> BaseAiCOCOImageOutputDataModel:

        output = self.formatter(model_out=model_out, images=images, categories=categories)
        return output


app = InferAPP(
    infer_function=SegmentationTritonInferenceModel(triton_url="triton:8000", model_name="toyseg"),
    input_data_model=BaseAiCOCOImageInputDataModel,
    output_data_model=BaseAiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
