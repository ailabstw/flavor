import os
import shutil
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from skops.hub_utils import download

from flavor.serve.apps import InferAPP
from flavor.serve.inference.data_models.api import (
    BaseAiCOCOTabularInputDataModel,
    BaseAiCOCOTabularOutputDataModel,
)
from flavor.serve.inference.data_models.functional import AiTable
from flavor.serve.inference.inference_models import BaseAiCOCOTabularInferenceModel
from flavor.serve.inference.strategies import AiCOCOTabularRegressionOutputStrategy

MODEL_PATH = os.path.join(os.getcwd(), str(uuid.uuid4()))


class RegressionInferenceModel(BaseAiCOCOTabularInferenceModel):
    def __init__(self):
        self.formatter = AiCOCOTabularRegressionOutputStrategy()
        super().__init__()

    def define_inference_network(self) -> Callable:
        download(repo_id="quantile-forest/california-housing-example", dst=MODEL_PATH)
        model = joblib.load(os.path.join(MODEL_PATH, "model.pkl"))
        return model

    def set_categories(self) -> List[Dict[str, Any]]:
        return None

    def set_regressions(self) -> None:
        regressions = [{"name": "reg_value"}]
        return regressions

    def data_reader(
        self, tables: Dict[str, Any], files: Sequence[str], **kwargs
    ) -> List[pd.DataFrame]:
        table_names = [table["file_name"].replace("/", "_") for table in tables]

        file_names = sorted(files, key=lambda s: s[::-1])
        table_names = sorted(table_names, key=lambda s: s[::-1])

        dataframes = []
        for file, table in zip(file_names, table_names):
            if not file.endswith(table):
                raise ValueError(f"File names do not match table names: {file} vs {table}")

            df = pd.read_csv(file)
            dataframes.append(df)

        return dataframes

    def preprocess(self, data: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(data)

    def inference(self, x: pd.DataFrame):
        out = self.network.predict(x).reshape(-1, 1)
        return out

    def postprocess(self, model_out: np.ndarray, **kwargs) -> np.ndarray:
        return model_out

    def output_formatter(
        self,
        model_out: Any,
        tables: Sequence[AiTable],
        dataframes: Sequence[pd.DataFrame],
        meta: Dict[str, Any],
        regressions: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs,
    ) -> BaseAiCOCOTabularOutputDataModel:

        output = self.formatter(
            model_out=model_out,
            tables=tables,
            dataframes=dataframes,
            regressions=regressions,
            meta=meta,
        )
        return output


app = InferAPP(
    infer_function=RegressionInferenceModel(),
    input_data_model=BaseAiCOCOTabularInputDataModel,
    output_data_model=BaseAiCOCOTabularOutputDataModel,
)

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 9111)))

    # remove downloaded files
    shutil.rmtree(MODEL_PATH, ignore_errors=True)
