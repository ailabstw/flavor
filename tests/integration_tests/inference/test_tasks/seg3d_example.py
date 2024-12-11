import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
import torch
from monai import transforms
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__()

    def define_inference_network(self) -> Callable:
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=14,
            feature_size=12,
            use_checkpoint=True,
        )
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.tiny_5000ep_f12_lr2e-4_pretrained.pt",
            progress=True,
            map_location=self.device,
        )["state_dict"]

        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        return model

    def set_categories(self) -> List[Dict[str, Any]]:
        categories = [
            {"name": "Background", "display": False},
            {"name": "Spleen", "display": True},
            {"name": "Right Kidney", "display": True},
            {"name": "Left Kidney", "display": True},
            {"name": "Gallbladder", "display": True},
            {"name": "Esophagus", "display": True},
            {"name": "Liver", "display": True},
            {"name": "Stomach", "display": True},
            {"name": "Aorta", "display": True},
            {"name": "IVC", "display": True},
            {"name": "Portal and Splenic Veins", "display": True},
            {"name": "Pancreas", "display": True},
            {"name": "Right adrenal gland", "display": True},
            {"name": "Left adrenal gland", "display": True},
        ]
        return categories

    def set_regressions(self) -> None:
        return None

    def data_reader(
        self, files: Sequence[str], **kwargs
    ) -> Tuple[np.ndarray, List[str], Tuple[int, ...]]:
        if len(files) > 1:
            # read multiple dicom
            def sort_images_by_z_axis(filenames):

                sorted_reader_filename_pairs = []

                for f in filenames:
                    dicom_reader = sitk.ImageFileReader()
                    dicom_reader.SetFileName(f)
                    dicom_reader.ReadImageInformation()

                    sorted_reader_filename_pairs.append((dicom_reader, f))

                zs = [
                    float(r.GetMetaData(key="0020|0032").split("\\")[-1])
                    for r, _ in sorted_reader_filename_pairs
                ]

                sort_inds = np.argsort(zs)
                sorted_reader_filename_pairs = [sorted_reader_filename_pairs[s] for s in sort_inds]

                return sorted_reader_filename_pairs

            pairs = sort_images_by_z_axis(files)

            readers, sorted_filenames = zip(*pairs)
            sorted_filenames = list(sorted_filenames)

            simages = [sitk.GetArrayFromImage(r.Execute()).squeeze() for r in readers]
            volume = np.stack(simages)
            volume = np.expand_dims(volume, axis=0)

            return volume, sorted_filenames, volume.shape[1:]

        else:
            # read nifti file
            sitk_reader = sitk.ReadImage(files)
            volume = sitk.GetArrayFromImage(sitk_reader)
            return volume, None, volume.shape[1:]

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        infer_transform = transforms.Compose(
            [
                transforms.Spacing(pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
                transforms.ScaleIntensityRange(
                    a_min=175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
                ),
                transforms.ToTensor(),
            ]
        )
        data = infer_transform(data).unsqueeze(0).to(self.device)

        return data

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = sliding_window_inference(
                x, (96, 96, 96), 4, self.network, overlap=0.5, mode="gaussian"
            )
        return out

    def postprocess(self, out: torch.Tensor, metadata: Tuple[int, ...]) -> np.ndarray:
        """
        Apply softmax and perform inverse resample back to original image size.

        Args:
            out (torch.Tensor): Inference model output.
            metadata (Tuple[int, ...]): Original image size.

        Returns:
            np.ndarray: Prediction output.
        """

        def resample_3d(img, target_size):
            imx, imy, imz = img.shape
            tx, ty, tz = target_size
            zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
            img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
            return img_resampled

        c = out.shape[1]
        output = torch.softmax(out, 1).cpu().numpy()
        output = np.argmax(output, axis=1).astype(np.uint8)[0]

        output = resample_3d(output, metadata)
        binary_output = np.zeros([c] + list(output.shape))
        for i in range(c):
            binary_output[i] = (output == i).astype(np.uint8)
        return binary_output

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
