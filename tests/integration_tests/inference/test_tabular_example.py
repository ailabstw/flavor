import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from httpx import ASGITransport, AsyncClient

from flavor.utils.infer_utils import compare_responses


@pytest.fixture
def configs():
    with open("tests/integration_tests/inference/test_configs.yaml", "r") as f:
        test_configs = yaml.load(f, Loader=yaml.SafeLoader)
    return test_configs["test_inference"]


class TestTabularInference:
    @pytest.fixture(autouse=True)
    def _set_configs(self, configs):
        self.configs = configs["test_tabular_inference"]

        self.target_fields = {
            "tables": (["id"], ["file_name"]),
            "records": (
                ["id"],
                ["table_id", "row_indexes"],
            ),
            "categories": (["id"], ["name"]),
            "regressions": (["id"], ["name"]),
        }
        torch.manual_seed(1999)
        np.random.seed(1999)

    def get_files_data(self, files_path, data_path):
        files = []
        files_path = Path(files_path)
        file = open(files_path, "rb")
        files.append(("files", (f"{files_path.name}", file)))

        with open(data_path, "r") as f:
            data = json.load(f)
        for k in data:
            data[k] = json.dumps(data[k])

        return files, data

    async def run_test(self, name: str, files_path: str, data_path: str):
        example = pytest.importorskip(f"test_tasks.{name}")
        example_app = example.app

        files, data = self.get_files_data(files_path, data_path)

        async with AsyncClient(
            transport=ASGITransport(app=example_app.app), base_url="http://test"
        ) as client:
            response = await client.post("/invocations", data=data, files=files)
            assert response.status_code == 200
            response_json = json.loads(response.text)

        with open(f"tests/integration_tests/inference/data/{name}.json", "r") as f:
            target = json.loads(f.read())

        identical = compare_responses(response_json, target, self.target_fields)
        assert identical

    @pytest.mark.asyncio
    async def test_tabular_cls(self):
        await self.run_test(name="tabular_cls_example", **self.configs["tabular_cls_example"])
