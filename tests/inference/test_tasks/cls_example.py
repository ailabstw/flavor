import os
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from flavor.serve.apps import InferAPP
from flavor.serve.inference import (
    BaseAiCOCOImageInferenceModel,
    BaseAiCOCOImageInputDataModel,
    BaseAiCOCOImageOutputDataModel,
)
from flavor.serve.models import AiImage, InferCategory
from flavor.serve.strategies import AiCOCOClassificationOutputStrategy


class ClassificationInferenceModel(BaseAiCOCOImageInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOClassificationOutputStrategy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def define_inference_network(self):
        network = resnet18(ResNet18_Weights.DEFAULT)
        network.eval()
        network.to(self.device)
        return network

    def set_categories(self):
        # ImageNet has 1000 categories
        categories = [{"name": str(i)} for i in range(1000)]
        return categories

    def set_regressions(self):
        return None

    def data_reader(self, files: Sequence[str], **kwargs) -> Tuple[np.ndarray, None, None]:
        img = Image.open(files[0])
        return img, None, None

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        transforms = ResNet18_Weights.DEFAULT.transforms()
        img = transforms(data).unsqueeze(0).to(self.device)
        return img

    def inference(self, x: Any) -> Any:
        with torch.no_grad():
            out = self.network(x)
        return out

    def postprocess(self, model_out: torch.Tensor, **kwargs) -> np.ndarray:
        model_out = model_out.squeeze(0).cpu().detach()
        model_out = (nn.functional.softmax(model_out, dim=0) > 0.4).long()
        return model_out.numpy()

    def output_formatter(
        self,
        model_out: np.ndarray,
        images: Sequence[AiImage],
        categories: List[InferCategory],
        **kwargs
    ) -> Any:

        output = self.formatter(model_out=model_out, images=images, categories=categories)
        return output


app = InferAPP(
    infer_function=ClassificationInferenceModel(),
    input_data_model=BaseAiCOCOImageInputDataModel,
    output_data_model=BaseAiCOCOImageOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))
