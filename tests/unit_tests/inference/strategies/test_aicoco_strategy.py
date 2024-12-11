import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flavor.serve.inference.data_models.functional import (
    AiAnnotation,
    AiCategory,
    AiImage,
    AiMeta,
    AiObject,
    AiRecord,
    AiRegression,
    AiTable,
    AiTableMeta,
)
from flavor.serve.inference.strategies import (
    AiCOCOClassificationOutputStrategy,
    AiCOCODetectionOutputStrategy,
    AiCOCORegressionOutputStrategy,
    AiCOCOSegmentationOutputStrategy,
    AiCOCOTabularClassificationOutputStrategy,
    AiCOCOTabularRegressionOutputStrategy,
)


class TestAiCOCOSegmentationOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCOSegmentationOutputStrategy()

    @pytest.fixture
    def sample_images(self):
        return [
            AiImage(
                id="img1", file_name="image1.dcm", index=0, category_ids=None, regressions=None
            ),
            AiImage(
                id="img2", file_name="image2.dcm", index=1, category_ids=None, regressions=None
            ),
        ]

    @pytest.fixture
    def sample_categories(self):
        return [
            {"name": "tumor1", "supercategory_name": "lesion"},
            {"name": "tumor2", "supercategory_name": "lesion"},
        ]

    @pytest.fixture
    def sample_model_output_semantic(self):
        model_output = np.zeros((2, 2, 4, 4))  # c, slice, h, w
        model_output[0, 0, :2, :2] = 1
        model_output[0, 1, 2, 1:] = 1
        model_output[1, 0, 1:3, :2] = 1
        model_output[1, 1, 1:3, :3] = 1
        return model_output

    @pytest.fixture
    def sample_model_output_instance(self):
        model_output = np.zeros((2, 2, 4, 4))  # c, slice, h, w
        model_output[0, 0, :2, :2] = 1
        model_output[0, 1, 2, 1:] = 2
        model_output[1, 0, 1:3, :2] = 3
        model_output[1, 1, 1:3, :3] = 4
        return model_output

    def test_prepare_aicoco(self, strategy, sample_images, sample_categories):
        result = strategy.prepare_aicoco(sample_images, sample_categories)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["images"]) == 2 and all(
            isinstance(image, AiImage) for image in result["images"]
        )
        assert len(result["categories"]) == len(sample_categories) + 1 and all(
            isinstance(category, AiCategory) for category in result["categories"]
        )  # 2 categories + 1 supercategory
        assert len(result["regressions"]) == 0
        assert isinstance(result["meta"], AiMeta)

    def test_validate_model_output_valid(self, strategy, sample_categories):
        valid_output = np.array([[[0, 1], [1, 0]], [[2, 0], [0, 0]]])
        strategy.validate_model_output(
            valid_output, sample_categories
        )  # Should not raise an exception

    def test_validate_model_output_invalid_type(self, strategy, sample_categories):
        invalid_output = [[0, 1], [1, 0]]  # Not a numpy array
        with pytest.raises(TypeError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"`model_out` must be type: np.ndarray but got {type(invalid_output)}."
        )

    def test_validate_model_output_invalid_shape(self, strategy, sample_categories):
        invalid_output = np.array([[[0, 1], [1, 0]]])  # Only one class
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"The number of classes in `model_out` should be {len(sample_categories)} but got {len(invalid_output)}."
        )

        invalid_output = np.array([[0, 1], [1, 0]])  # dim == 2
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"The dimension of `model_out` should be in 3D or 4D but got {invalid_output.ndim}."
        )

    def test_validate_model_output_invalid_values(self, strategy, sample_categories):
        invalid_output = np.array([[[0.5, 1.5], [1, 0]], [[1, 0], [0, 1]]])  # Non-integer values
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == "The value of `model_out` should be integer such as 0, 1, 2 ... with int or float type."
        )

    def check_annotations_objects(self, result, expected_output):
        # Compare the segmentations
        for actual_ann, expected_ann in zip(result["annotations"], expected_output["annotations"]):
            assert (
                actual_ann.segmentation == expected_ann["segmentation"]
            ), "Segmentation in annotation does not match the expected output"

        # Compare the category_ids of objects
        for actual_obj, expected_obj in zip(result["objects"], expected_output["objects"]):
            assert (
                actual_obj.category_ids == expected_obj["category_ids"]
            ), "Category IDs in object do not match the expected output"

    def test_generate_annotations_objects(
        self, strategy, sample_model_output_semantic, sample_model_output_instance
    ):
        # Set up the strategy with necessary attributes
        strategy.aicoco_categories = [
            AiCategory(id="c1", name="tumor1", supercategory_id="sc1"),
            AiCategory(id="c2", name="tumor2", supercategory_id="sc1"),
        ]
        strategy.images_ids = ["img1", "img2"]

        semantic_seg_result = strategy.generate_annotations_objects(sample_model_output_semantic)
        instance_seg_result = strategy.generate_annotations_objects(sample_model_output_instance)
        semantic_seg_result_w_display = strategy.generate_annotations_objects(
            sample_model_output_semantic
        )

        src_path = Path(__file__).parent / "data"
        with open(src_path / "aicoco_semantic_seg.json", "r") as f:
            expected_output_semantic = json.load(f)
        with open(src_path / "aicoco_instance_seg.json", "r") as f:
            expected_output_instance = json.load(f)
        with open(src_path / "aicoco_semantic_seg_w_display.json", "r") as f:
            expected_output_semantic_w_display = json.load(f)
        self.check_annotations_objects(semantic_seg_result, expected_output_semantic)
        self.check_annotations_objects(instance_seg_result, expected_output_instance)
        self.check_annotations_objects(
            semantic_seg_result_w_display, expected_output_semantic_w_display
        )

    def test_model_to_aicoco(
        self, strategy, sample_images, sample_categories, sample_model_output_semantic
    ):
        aicoco_ref = strategy.prepare_aicoco(sample_images, sample_categories)
        result = strategy.model_to_aicoco(aicoco_ref, sample_model_output_semantic)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result
        assert "regressions" in result

    def test_call_method(
        self, strategy, sample_images, sample_categories, sample_model_output_semantic
    ):
        result = strategy(sample_model_output_semantic, sample_images, sample_categories)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result
        assert "regressions" in result


class TestAiCOCOClassificationOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCOClassificationOutputStrategy()

    @pytest.fixture
    def sample_images(self):
        return [
            AiImage(id="img1", file_name="image1.jpg", index=0, category_ids=None, regressions=None)
        ]

    @pytest.fixture
    def sample_3d_images(self):
        return [
            AiImage(
                id="img1", file_name="image1.jpg", index=0, category_ids=None, regressions=None
            ),
            AiImage(
                id="img2", file_name="image2.jpg", index=1, category_ids=None, regressions=None
            ),
        ]

    @pytest.fixture
    def sample_categories(self):
        return [
            {"name": "tumor1", "supercategory_name": "lesion"},
            {"name": "tumor2", "supercategory_name": "lesion"},
        ]

    @pytest.fixture
    def sample_model_output(self):
        return np.array([1, 0])  # tumor1 is present, tumor2 is not

    def test_prepare_aicoco(self, strategy, sample_images, sample_categories):
        result = strategy.prepare_aicoco(sample_images, categories=sample_categories)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["images"]) == 1 and all(
            isinstance(image, AiImage) for image in result["images"]
        )
        assert len(result["categories"]) == len(sample_categories) + 1 and all(
            isinstance(category, AiCategory) for category in result["categories"]
        )  # 2 categories + 1 supercategory
        assert len(result["regressions"]) == 0
        assert isinstance(result["meta"], AiMeta)

    def test_validate_model_output_valid(self, strategy, sample_categories):
        valid_output = np.array([1, 0])
        strategy.validate_model_output(
            valid_output, sample_categories
        )  # Should not raise an exception

    def test_validate_model_output_invalid_type(self, strategy, sample_categories):
        invalid_output = [1, 0]  # Not a numpy array
        with pytest.raises(TypeError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"`model_out` must be type: np.ndarray but got {type(invalid_output)}."
        )

    def test_validate_model_output_invalid_shape(self, strategy, sample_categories):
        invalid_output = np.array([1])  # Only one class
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"The number of classes in `model_out` should be {len(sample_categories)} but got {len(invalid_output)}."
        )

        invalid_output = np.array([[1, 0], [0, 1]])  # 2D array instead of 1D
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"The dimension of `model_out` should be in 1D but got {invalid_output.ndim}."
        )

    def test_validate_model_output_invalid_values(self, strategy, sample_categories):
        invalid_output = np.array([0.5, 1.5])  # Non-binary values
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == "The value of `model_out` should be only 0 or 1 with int or float type."
        )

    def test_update_images_meta_2d(self, strategy, sample_images, sample_model_output):
        strategy.aicoco_categories = [
            AiCategory(id="c1", name="tumor1", supercategory_id="sc1"),
            AiCategory(id="c2", name="tumor2", supercategory_id="sc1"),
        ]
        meta = AiMeta(category_ids=None, regressions=None)

        updated_images, updated_meta = strategy.update_images_meta(
            sample_model_output, sample_images, meta
        )

        assert len(updated_images) == 1
        assert updated_images[0].category_ids == ["c1"]
        assert updated_meta.category_ids is None

    def test_update_images_meta_3d(self, strategy, sample_3d_images, sample_model_output):
        strategy.aicoco_categories = [
            AiCategory(id="c1", name="tumor1", supercategory_id="sc1"),
            AiCategory(id="c2", name="tumor2", supercategory_id="sc1"),
        ]
        meta = AiMeta(category_ids=None, regressions=None)

        updated_images, updated_meta = strategy.update_images_meta(
            sample_model_output, sample_3d_images, meta
        )

        assert len(updated_images) == 2
        assert all(img.category_ids is None for img in updated_images)
        assert updated_meta.category_ids == ["c1"]

    def test_model_to_aicoco(self, strategy, sample_images, sample_model_output):
        aicoco_categories = [
            AiCategory(id="c1", name="tumor1", supercategory_id="sc1"),
            AiCategory(id="c2", name="tumor2", supercategory_id="sc1"),
        ]
        aicoco_ref = {
            "images": sample_images,
            "categories": aicoco_categories,
            "regressions": [],
            "meta": AiMeta(category_ids=None, regressions=None),
        }
        strategy.aicoco_categories = aicoco_categories
        result = strategy.model_to_aicoco(aicoco_ref, sample_model_output)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result
        assert "regressions" in result

        assert len(result["annotations"]) == 0
        assert len(result["objects"]) == 0
        assert result["images"][0].category_ids == ["c1"]

    def test_call_method(self, strategy, sample_images, sample_categories, sample_model_output):
        result = strategy(sample_model_output, sample_images, sample_categories)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result
        assert "regressions" in result

        assert len(result["annotations"]) == 0
        assert len(result["objects"]) == 0


class TestAiCOCODetectionOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCODetectionOutputStrategy()

    @pytest.fixture
    def sample_images(self):
        return [
            AiImage(id="img1", file_name="image1.dcm", index=0, category_ids=None, regressions=None)
        ]

    @pytest.fixture
    def sample_categories(self):
        return [
            {"name": "tumor1", "supercategory_name": "lesion"},
            {"name": "tumor2", "supercategory_name": "lesion"},
        ]

    @pytest.fixture
    def sample_regressions(self):
        return [
            {"name": "density", "superregression_name": "property"},
            {"name": "severity", "superregression_name": "property"},
        ]

    def test_prepare_aicoco(self, strategy, sample_images, sample_categories, sample_regressions):
        result = strategy.prepare_aicoco(
            sample_images, categories=sample_categories, regressions=sample_regressions
        )

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["images"]) == 1 and all(
            isinstance(image, AiImage) for image in result["images"]
        )
        assert len(result["categories"]) == len(sample_categories) + 1 and all(
            isinstance(category, AiCategory) for category in result["categories"]
        )  # 2 categories + 1 supercategory
        assert len(result["regressions"]) == len(sample_regressions) + 1 and all(
            isinstance(regression, AiRegression) for regression in result["regressions"]
        )  # 2 regressions + 1 superregression
        assert isinstance(result["meta"], AiMeta)

    def test_validate_model_output_valid(self, strategy, sample_categories):
        valid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[1, 0], [0, 1]]),
            "confidence_score": np.array([0.9, 0.8]),
            "regression_value": np.array([[5, 10], [3, 8]]),
        }

        # This should not raise any exception
        strategy.validate_model_output(valid_output, sample_categories)

    def test_validate_model_output_invalid_elements(self, strategy, sample_categories):
        # test the amoubt of `cls_pred`
        invalid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10]]),
            "cls_pred": np.array([[1, 0], [0, 1]]),
        }
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value) == "`bbox_pred` and `cls_pred` should have same amount of elements."
        )

        # test the amoubt of `confidence_score`
        invalid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10]]),
            "cls_pred": np.array([[1, 0]]),
            "confidence_score": np.array([0.9, 0.8]),
        }
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == "`bbox_pred` and `confidence_score` should have same amount of elements."
        )

        # test the amoubt of `regression_value`
        invalid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10]]),
            "cls_pred": np.array([[1, 0]]),
            "regression_value": np.array([[5, 10], [3, 8]]),
        }
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == "`bbox_pred` and `regression_value` should have same amount of elements."
        )

    def test_validate_model_output_invalid_cls_shape(self, strategy, sample_categories):
        invalid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[1, 0, 0], [0, 1, 0]]),
        }
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == f"The length of each element in `cls_pred` should be {len(sample_categories)}."
        )

    def test_validate_model_output_invalid_values(self, strategy, sample_categories):
        invalid_output = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[0.5, 1.5], [0, 1]]),  # Non-binary values
        }
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_categories)
        assert (
            str(exc_info.value)
            == "The value of `cls_pred` should be only 0 or 1 with int or float type."
        )

    def test_generate_annotations_objects(self, strategy):
        strategy.images_ids = ["img1"]
        strategy.aicoco_categories = [
            AiCategory(id="c1", name="tumor1", supercategory_id="sc1"),
            AiCategory(id="c2", name="tumor2", supercategory_id="sc1"),
        ]
        strategy.aicoco_regressions = [
            AiRegression(id="r1", name="density", superregression_id="sr1"),
            AiRegression(id="r2", name="severity", superregression_id="sr1"),
        ]

        model_out = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[1, 0], [0, 1]]),
            "confidence_score": np.array([0.9, 0.8]),
            "regressions": np.array([[5, 10], [3, 8]]),
        }

        result = strategy.generate_annotations_objects(model_out)

        assert "annotations" in result
        assert "objects" in result
        assert len(result["annotations"]) == 2
        assert len(result["objects"]) == 2

        assert isinstance(result["annotations"][0], AiAnnotation)
        assert isinstance(result["objects"][0], AiObject)

    def test_model_to_aicoco(self, strategy, sample_images, sample_categories, sample_regressions):
        aicoco_ref = strategy.prepare_aicoco(sample_images, sample_categories, sample_regressions)

        model_out = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[1, 0], [0, 1]]),
            "confidence_score": np.array([0.9, 0.8]),
            "regression": np.array([[5, 10], [3, 8]]),
        }

        result = strategy.model_to_aicoco(aicoco_ref, model_out)

        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result

    def test_call_method(self, strategy, sample_images, sample_categories, sample_regressions):
        model_out = {
            "bbox_pred": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            "cls_pred": np.array([[1, 0], [0, 1]]),
            "confidence_score": np.array([0.9, 0.8]),
            "regression": np.array([[5, 10], [3, 8]]),
        }

        result = strategy(model_out, sample_images, sample_categories, sample_regressions)

        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result


class TestAiCOCORegressionOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCORegressionOutputStrategy()

    @pytest.fixture
    def sample_images(self):
        return [
            AiImage(id="img1", file_name="image1.dcm", index=0, category_ids=None, regressions=None)
        ]

    @pytest.fixture
    def sample_3d_images(self):
        return [
            AiImage(
                id="img1", file_name="image1.jpg", index=0, category_ids=None, regressions=None
            ),
            AiImage(
                id="img2", file_name="image2.jpg", index=1, category_ids=None, regressions=None
            ),
        ]

    @pytest.fixture
    def sample_regressions(self):
        return [
            {"name": "density", "superregression_name": "property"},
            {"name": "severity", "superregression_name": "property"},
        ]

    @pytest.fixture
    def sample_model_output(self):
        return np.array([25.0, 70.0])

    def test_prepare_aicoco(self, strategy, sample_images, sample_regressions):
        result = strategy.prepare_aicoco(sample_images, regressions=sample_regressions)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["images"]) == 1 and all(
            isinstance(image, AiImage) for image in result["images"]
        )
        assert len(result["categories"]) == 0

        assert len(result["regressions"]) == len(sample_regressions) + 1 and all(
            isinstance(regression, AiRegression) for regression in result["regressions"]
        )  # 2 regressions + 1 superregression
        assert isinstance(result["meta"], AiMeta)

    def test_validate_model_output_valid(self, strategy, sample_regressions):
        valid_output_1d = np.array([1.0, 2.0])

        # These should not raise any exception
        strategy.validate_model_output(valid_output_1d, sample_regressions)

    def test_validate_model_output_invalid_type(self, strategy, sample_regressions):
        invalid_output = [1.0, 2.0]

        with pytest.raises(TypeError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_regressions)
        assert (
            str(exc_info.value)
            == f"`model_out` must be type: np.ndarray but got {type(invalid_output)}."
        )

    def test_validate_model_output_invalid_shape(self, strategy, sample_regressions):
        invalid_output = np.array([1])  # Only one class
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_regressions)
        assert (
            str(exc_info.value)
            == f"The number of regression values in `model_out` should be {len(sample_regressions)} but got {len(invalid_output)}."
        )

        invalid_output = np.random.rand(2, 2)  # extra dimension
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(invalid_output, sample_regressions)
        assert (
            str(exc_info.value)
            == f"The dimension of `model_out` should be in 1D but got {invalid_output.ndim}."
        )

    def test_update_images_meta_2d(self, strategy, sample_images, sample_model_output):
        strategy.aicoco_regressions = [
            AiRegression(id="r1", name="age", superregression_id="sr1"),
            AiRegression(id="r2", name="severity", superregression_id="sr1"),
        ]
        meta = AiMeta(category_ids=None, regressions=None)

        updated_images, updated_meta = strategy.update_images_meta(
            sample_model_output, sample_images, meta
        )

        assert len(updated_images) == 1
        assert updated_images[0].regressions[0].value == 25.0
        assert updated_images[0].regressions[1].value == 70.0
        assert updated_meta.regressions is None

    def test_update_images_meta_3d(self, strategy, sample_3d_images, sample_model_output):
        strategy.aicoco_regressions = [
            AiRegression(id="r1", name="age", superregression_id="sr1"),
            AiRegression(id="r2", name="severity", superregression_id="sr1"),
        ]

        meta = AiMeta(category_ids=None, regressions=None)

        updated_images, updated_meta = strategy.update_images_meta(
            sample_model_output, sample_3d_images, meta
        )

        assert len(updated_images) == 2
        assert all(image.regressions is None for image in updated_images)
        assert updated_meta.regressions[0].value == 25.0
        assert updated_meta.regressions[1].value == 70.0

    def test_model_to_aicoco(self, strategy, sample_images, sample_model_output):
        aicoco_regressions = [
            AiRegression(id="r1", name="age", superregression_id="sr1"),
            AiRegression(id="r2", name="severity", superregression_id="sr1"),
        ]
        aicoco_ref = {
            "images": sample_images,
            "categories": [],
            "regressions": aicoco_regressions,
            "meta": AiMeta(category_ids=None, regressions=None),
        }
        strategy.aicoco_regressions = aicoco_regressions
        result = strategy.model_to_aicoco(aicoco_ref, sample_model_output)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result

        assert result["images"][0].regressions is not None
        assert len(result["images"][0].regressions) == 2

    def test_call_method(self, strategy, sample_images, sample_regressions, sample_model_output):
        result = strategy(sample_model_output, sample_images, regressions=sample_regressions)

        assert isinstance(result, dict)
        assert "images" in result
        assert "categories" in result
        assert "regressions" in result
        assert "annotations" in result
        assert "objects" in result
        assert "meta" in result

        assert len(result["annotations"]) == 0
        assert len(result["objects"]) == 0


class TestAiCOCOTabularClassificationOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCOTabularClassificationOutputStrategy()

    @pytest.fixture
    def sample_tables(self):
        return [{"id": "tb1", "file_name": "test_table"}]

    @pytest.fixture
    def sample_dataframes(self):
        return [pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})]

    @pytest.fixture
    def sample_categories(self):
        return [
            {"name": "symptom1", "supercategory_name": "disease"},
            {"name": "symptom2", "supercategory_name": "disease"},
        ]

    @pytest.fixture
    def sample_meta(self):
        return {"window_size": 1}

    @pytest.fixture
    def sample_model_output(self):
        return np.array([[1, 0], [0, 1], [1, 1]])

    def test_prepare_aicoco(
        self, strategy, sample_tables, sample_dataframes, sample_categories, sample_meta
    ):
        result = strategy.prepare_aicoco(
            tables=sample_tables,
            meta=sample_meta,
            dataframes=sample_dataframes,
            categories=sample_categories,
        )

        assert isinstance(result, dict)
        assert "tables" in result
        assert "records" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["tables"]) == len(sample_tables) and all(
            isinstance(table, AiTable) for table in result["tables"]
        )
        assert len(result["records"]) == (
            len(sample_dataframes[0]) // sample_meta["window_size"]
        ) and all(isinstance(record, AiRecord) for record in result["records"])
        assert len(result["categories"]) == len(sample_categories) + 1 and all(
            isinstance(category, AiCategory) for category in result["categories"]
        )  # 2 categories + 1 supercategory
        assert len(result["regressions"]) == 0
        assert isinstance(result["meta"], AiTableMeta)

    def test_validate_model_output_valid(
        self, strategy, sample_categories, sample_dataframes, sample_meta
    ):
        valid_output = np.array([[1, 0], [0, 1], [1, 1]])
        strategy.validate_model_output(
            valid_output, sample_categories, sample_dataframes, sample_meta
        )

    def test_validate_model_output_invalid_type(
        self, strategy, sample_categories, sample_dataframes, sample_meta
    ):
        invalid_output = [[1, 0], [0, 1], [1, 1]]
        with pytest.raises(TypeError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_categories, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"`model_out` must be type: np.ndarray but got {type(invalid_output)}."
        )

    def test_validate_model_output_invalid_shape(
        self, strategy, sample_categories, sample_dataframes, sample_meta
    ):
        invalid_output = np.array([[[1, 0], [0, 1], [0, 1]]])  # 3D array instead of 2D
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_categories, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The dimension of the `model_out` must be 2 but got {invalid_output.ndim}."
        )

        invalid_output = np.array([[1, 0], [0, 1]])  # missing records
        n = sum([len(df) // sample_meta["window_size"] for df in sample_dataframes])
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_categories, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The number of records in `model_out` should be {n} but got {invalid_output.shape[0]}."
        )

        invalid_output = np.array([[1], [0], [1]])  # Only one class
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_categories, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The number of classes in `model_out` should be {len(sample_categories)} but got {invalid_output.shape[-1]}."
        )

    def test_validate_model_output_invalid_values(
        self, strategy, sample_categories, sample_dataframes, sample_meta
    ):
        invalid_output = np.array([[1, 0], [0, 1], [0.5, 1.5]])  # Non-binary values
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_categories, sample_dataframes, sample_meta
            )
        assert str(exc_info.value) == "`model_out` contains elements other than 0 and 1"

    def test_model_to_aicoco(self, strategy, sample_tables, sample_meta, sample_model_output):
        aicoco_categories = [
            AiCategory(id="c1", name="symptom1", supercategory_id="sc1"),
            AiCategory(id="c2", name="symptom2", supercategory_id="sc1"),
        ]
        aicoco_records = [
            AiRecord(
                id="r1",
                table_id="tb1",
                row_indexes=list(
                    range(i * sample_meta["window_size"], (i + 1) * sample_meta["window_size"])
                ),
                regressions=None,
                category_ids=None,
            )
            for i in range(3)
        ]
        aicoco_ref = {
            "tables": sample_tables,
            "categories": aicoco_categories,
            "regressions": [],
            "records": aicoco_records,
            "meta": sample_meta,
        }
        strategy.aicoco_categories = aicoco_categories
        strategy.aicoco_records = aicoco_records

        result = strategy.model_to_aicoco(aicoco_ref, sample_model_output)

        assert len(result["records"]) == 3
        assert result["records"][0].category_ids == [aicoco_categories[0].id]
        assert result["records"][1].category_ids == [aicoco_categories[1].id]
        assert set(result["records"][2].category_ids) == {
            aicoco_categories[0].id,
            aicoco_categories[1].id,
        }

    def test_call_method(
        self,
        strategy,
        sample_tables,
        sample_dataframes,
        sample_categories,
        sample_meta,
        sample_model_output,
    ):
        result = strategy(
            sample_model_output, sample_tables, sample_dataframes, sample_categories, sample_meta
        )

        assert isinstance(result, dict)
        assert "tables" in result
        assert "categories" in result
        assert "regressions" in result
        assert "records" in result
        assert "meta" in result
        assert len(result["records"]) == 3
        assert all(record.category_ids for record in result["records"])


class TestAiCOCOTabularRegressionOutputStrategy:
    @pytest.fixture
    def strategy(self):
        return AiCOCOTabularRegressionOutputStrategy()

    @pytest.fixture
    def sample_tables(self):
        return [{"id": "tb1", "file_name": "test_table"}]

    @pytest.fixture
    def sample_dataframes(self):
        return [pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})]

    @pytest.fixture
    def sample_regressions(self):
        return [
            {"name": "density", "superregression_name": "property"},
            {"name": "severity", "superregression_name": "property"},
        ]

    @pytest.fixture
    def sample_meta(self):
        return {"window_size": 1}

    @pytest.fixture
    def sample_model_output(self):
        return np.array([[25, 70], [30, 80], [35, 90]])

    def test_prepare_aicoco(
        self, strategy, sample_tables, sample_dataframes, sample_regressions, sample_meta
    ):
        result = strategy.prepare_aicoco(
            tables=sample_tables,
            meta=sample_meta,
            dataframes=sample_dataframes,
            regressions=sample_regressions,
        )

        assert isinstance(result, dict)
        assert "tables" in result
        assert "records" in result
        assert "categories" in result
        assert "regressions" in result
        assert "meta" in result

        assert len(result["tables"]) == len(sample_tables) and all(
            isinstance(table, AiTable) for table in result["tables"]
        )
        assert len(result["records"]) == (
            len(sample_dataframes[0]) // sample_meta["window_size"]
        ) and all(isinstance(record, AiRecord) for record in result["records"])
        assert len(result["categories"]) == 0
        assert len(result["regressions"]) == len(sample_regressions) + 1 and all(
            isinstance(regression, AiRegression) for regression in result["regressions"]
        )  # 2 regressions + 1 superregression
        assert isinstance(result["meta"], AiTableMeta)

    def test_validate_model_output_valid(
        self, strategy, sample_regressions, sample_dataframes, sample_meta
    ):
        valid_output = np.array([[25, 70], [30, 80], [35, 90]])

        strategy.validate_model_output(
            valid_output, sample_regressions, sample_dataframes, sample_meta
        )

    def test_validate_model_output_invalid_type(
        self, strategy, sample_regressions, sample_dataframes, sample_meta
    ):
        invalid_output = [[25, 70], [30, 80], [35, 90]]
        with pytest.raises(TypeError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_regressions, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"`model_out` must be type: np.ndarray but got {type(invalid_output)}."
        )

    def test_validate_model_output_invalid_shape(
        self, strategy, sample_regressions, sample_dataframes, sample_meta
    ):
        invalid_output = np.array([[[25, 70], [30, 80], [35, 90]]])  # 3D array instead of 2D
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_regressions, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The dimension of the `model_out` must be 2 but got {invalid_output.ndim}."
        )

        invalid_output = np.array([[1, 0], [0, 1]])  # missing records
        n = sum([len(df) // sample_meta["window_size"] for df in sample_dataframes])
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_regressions, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The number of records in `model_out` should be {n} but got {invalid_output.shape[0]}."
        )

        invalid_output = np.array([[1], [0], [1]])  # Only one class
        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_regressions, sample_dataframes, sample_meta
            )
        assert (
            str(exc_info.value)
            == f"The number of regression values in `model_out` should be {len(sample_regressions)} but got {invalid_output.shape[-1]}."
        )

    def test_validate_model_output_invalid_values(
        self, strategy, sample_regressions, sample_dataframes, sample_meta
    ):
        invalid_output = np.array([[1, np.inf], [3, 4], [5, 6]])

        with pytest.raises(ValueError) as exc_info:
            strategy.validate_model_output(
                invalid_output, sample_regressions, sample_dataframes, sample_meta
            )
        assert str(exc_info.value) == "The value of `model_out` should not contain finite numbers."

    def test_model_to_aicoco(self, strategy, sample_tables, sample_meta, sample_model_output):
        aicoco_regressions = [
            AiRegression(id="r1", name="age", superregression_id="sr1"),
            AiRegression(id="r2", name="severity", superregression_id="sr1"),
        ]
        aicoco_records = [
            AiRecord(
                id="r1",
                table_id="tb1",
                row_indexes=list(
                    range(i * sample_meta["window_size"], (i + 1) * sample_meta["window_size"])
                ),
                regressions=None,
                category_ids=None,
            )
            for i in range(3)
        ]

        aicoco_ref = {
            "tables": sample_tables,
            "categories": [],
            "regressions": aicoco_regressions,
            "records": aicoco_records,
            "meta": sample_meta,
        }

        strategy.aicoco_regressions = aicoco_regressions
        strategy.aicoco_records = aicoco_records

        result = strategy.model_to_aicoco(aicoco_ref, sample_model_output)

        assert len(result["records"]) == 3
        for i, record in enumerate(result["records"]):
            assert len(record.regressions) == 2
            assert record.regressions[0].regression_id == aicoco_regressions[0].id
            assert record.regressions[0].value == sample_model_output[i][0]
            assert record.regressions[1].regression_id == aicoco_regressions[1].id
            assert record.regressions[1].value == sample_model_output[i][1]

    def test_call_method(
        self,
        strategy,
        sample_tables,
        sample_dataframes,
        sample_regressions,
        sample_meta,
        sample_model_output,
    ):

        result = strategy(
            sample_model_output, sample_tables, sample_dataframes, sample_regressions, sample_meta
        )

        assert isinstance(result, dict)
        assert "tables" in result
        assert "categories" in result
        assert "regressions" in result
        assert "records" in result
        assert "meta" in result
        assert len(result["records"]) == 3
        assert all(record.regressions for record in result["records"])
        assert all(len(record.regressions) == 2 for record in result["records"])
