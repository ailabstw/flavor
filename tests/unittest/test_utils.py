import os
import unittest
from multiprocessing import Value
from unittest.mock import mock_open, patch

# Import the module containing the functions
from flavor.cook.utils import (
    CleanEvent,
    CleanInfoJson,
    GetPaths,
    SaveInfoJson,
    SetEvent,
    SetPaths,
    WaitEvent,
    compareVersion,
)


class TestUtilityFunctions(unittest.TestCase):
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    @patch("builtins.open", new_callable=mock_open)
    def test_SetEvent_success(self, mock_file):
        event = "TrainStarted"
        SetEvent(event)
        mock_file.assert_called_with("/tmp/TrainStarted", "w")

    @patch.dict(os.environ, {})
    def test_SetEvent_missing_env(self):
        event = "TrainStarted"
        with self.assertRaises(EnvironmentError) as context:
            SetEvent(event)
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    def test_SetEvent_unknown_event(self):
        event = "UnknownEvent"
        with self.assertRaises(ValueError):
            SetEvent(event)

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_CleanEvent_success(self, mock_remove, mock_exists):
        event = "TrainStarted"
        CleanEvent(event)
        mock_remove.assert_called_with("/tmp/TrainStarted")

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_CleanEvent_file_not_exists(self, mock_remove, mock_exists):
        event = "TrainStarted"
        CleanEvent(event)
        mock_remove.assert_not_called()

    @patch("os.path.exists", side_effect=[False, False, True])
    @patch("os.remove")
    @patch("time.sleep", return_value=None)
    def test_WaitEvent_no_error(self, mock_sleep, mock_remove, mock_exists):
        is_error = Value("i", 0)
        WaitEvent("event1", is_error)
        self.assertEqual(mock_exists.call_count, 3)
        mock_remove.assert_called_once_with(os.path.join(self.output_path, "event1"))
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch("time.sleep", return_value=None)
    def test_WaitEvent_with_error(self, mock_sleep, mock_remove, mock_exists):
        is_error = Value("i", 1)
        WaitEvent("event1", is_error)
        self.assertEqual(mock_exists.call_count, 1)
        mock_remove.assert_not_called()
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("os.environ.get", return_value=None)
    def test_WaitEvent_missing_env(self, mock_env_get):
        event = "TrainStarted"
        with self.assertRaises(EnvironmentError) as context:
            WaitEvent(event)
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_SaveInfoJson_success(self, mock_file):
        info = {"key": "value"}
        SaveInfoJson(info)
        mock_file.assert_called_with("/tmp/info.json", "w")

    @patch("os.environ.get", return_value=None)
    def test_SaveInfoJson_missing_env(self, mock_env_get):
        info = {"key": "value"}
        with self.assertRaises(EnvironmentError) as context:
            SaveInfoJson(info)
        self.assertEqual(
            str(context.exception), "LOCAL_MODEL_PATH environment variable is not set."
        )

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_CleanInfoJson_success(self, mock_remove, mock_exists):
        CleanInfoJson()
        mock_remove.assert_called_with("/tmp/info.json")

    @patch("os.path.exists", return_value=False)
    @patch("os.remove")
    @patch.dict(os.environ, {"LOCAL_MODEL_PATH": "/tmp/model.pt"})
    def test_CleanInfoJson_file_not_exists(self, mock_remove, mock_exists):
        CleanInfoJson()
        mock_remove.assert_not_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_SetPaths_with_string(self, mock_file):
        SetPaths("localModels", "model1.pt")
        mock_file.assert_called_with("/tmp/localModels", "w")
        mock_file().write.assert_called_with("model1.pt")

    @patch("builtins.open", new_callable=mock_open)
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_SetPaths_with_list(self, mock_file):
        SetPaths("localModels", ["model1.pt", "model2.pt"])
        mock_file.assert_called_with("/tmp/localModels", "w")
        mock_file().write.assert_called_with("model1.pt\nmodel2.pt")

    def test_SetPaths_unknown_filename(self):
        with self.assertRaises(ValueError):
            SetPaths("unknownFile", "data")

    @patch("builtins.open", new_callable=mock_open, read_data="model1.pt\nmodel2.pt")
    @patch.dict(os.environ, {"OUTPUT_PATH": "/tmp"})
    def test_GetPaths_success(self, mock_file):
        result = GetPaths("localModels")
        self.assertEqual(result, ["model1.pt", "model2.pt"])

    @patch.dict(os.environ, {}, clear=True)
    def test_GetPaths_missing_env(self):
        with self.assertRaises(EnvironmentError) as context:
            GetPaths("localModels")
        self.assertEqual(str(context.exception), "OUTPUT_PATH environment variable is not set.")

    def test_GetPaths_unknown_filename(self):
        with self.assertRaises(ValueError):
            GetPaths("unknownFile")

    def test_compareVersion_equal(self):
        result = compareVersion("1.0.0", "1.0.0")
        self.assertEqual(result, 0)

    def test_compareVersion_v1_less(self):
        result = compareVersion("1.0.0", "1.0.1")
        self.assertEqual(result, -1)

    def test_compareVersion_v1_greater(self):
        result = compareVersion("1.0.1", "1.0.0")
        self.assertEqual(result, 1)

    def test_compareVersion_non_integer(self):
        with self.assertRaises(ValueError):
            compareVersion("1.0a", "1.0.0")


if __name__ == "__main__":
    unittest.main()
