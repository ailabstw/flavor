import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    BaseAiCOCOImageInputDataModel,
    BaseAiCOCOImageOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiImage
from flavor.serve.inference.inference_models import BaseAiCOCOImageInferenceModel
from flavor.serve.inference.strategies import AiCOCOSegmentationOutputStrategy


class SegmentationInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOSegmentationOutputStrategy()
        super().__init__()

    def define_inference_network(self) -> Callable:
        return LMInferer(modelname="LTRCLobes", fillmodel="R231")

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [
            {"name": "Background", "display": False},
            {"name": "Left Upper Lobe", "display": True},
            {"name": "Left Lower Lobe", "display": True},
            {"name": "Right Upper Lobe", "display": True},
            {"name": "Right Middle Lobe", "display": True},
            {"name": "Right Lower Lobe", "display": True},
        ]
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        dicom_reader = sitk.ImageFileReader()
        dicom_reader.SetFileName(files[0])
        dicom_reader.ReadImageInformation()
        dicom = sitk.GetArrayFromImage(dicom_reader.Execute()).squeeze()

        data = np.expand_dims(dicom, axis=0)
        return data, None, None

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        return data

    def inference(self, x: np.ndarray) -> np.ndarray:
        return self.network.apply(x)

    def postprocess(self, out: Any, metadata: Optional[Any] = None) -> np.ndarray:
        # (1, h, w) -> (c, h, w)
        out = [
            np.expand_dims((out == i).astype(np.uint8), axis=0)
            for i in range(6)  # or len(self.categories)
        ]
        out = np.concatenate(out, axis=0)
        return out

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
    infer_function=SegmentationInferenceModel(),
    input_data_model=BaseAiCOCOImageInputDataModel,
    output_data_model=BaseAiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
