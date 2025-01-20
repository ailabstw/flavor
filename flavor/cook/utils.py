import json
import multiprocessing as mp
import os
import time
from itertools import zip_longest
from typing import Union

EventSet = {
    "TrainInitDone",
    "TrainStarted",
    "TrainFinished",
    "AggregateStarted",
    "AggregateFinished",
    "ProcessFinished",
}
PathSet = {"localModels", "localInfos", "globalModel", "globalInfo"}


def SetEvent(event: str):
    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))
    output_path = os.environ.get("OUTPUT_PATH")
    if not output_path:
        raise EnvironmentError("OUTPUT_PATH environment variable is not set.")
    open(os.path.join(output_path, event), "w").close()


def WaitEvent(event: str, is_error: mp.Value = None):

    if is_error is None:
        is_error = mp.Value("i", 0)

    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))

    output_path = os.environ.get("OUTPUT_PATH")
    if not output_path:
        raise EnvironmentError("OUTPUT_PATH environment variable is not set.")

    while not os.path.exists(os.path.join(output_path, event)):
        time.sleep(1)
        if is_error.value != 0:
            return

    os.remove(os.path.join(output_path, event))


def CleanEvent(event: str):
    if event in EventSet and os.path.exists(os.path.join(os.environ["OUTPUT_PATH"], event)):
        os.remove(os.path.join(os.environ["OUTPUT_PATH"], event))


def CleanAllEvent():
    for event in EventSet:
        if os.path.exists(os.path.join(os.environ["OUTPUT_PATH"], event)):
            os.remove(os.path.join(os.environ["OUTPUT_PATH"], event))


def IsSetEvent(event: str) -> bool:
    if event not in EventSet:
        raise ValueError("Unknown event {}".format(event))
    return os.path.exists(os.path.join(os.environ["OUTPUT_PATH"], event))


def SaveInfoJson(info: dict):
    local_model_path = os.environ.get("LOCAL_MODEL_PATH")
    if not local_model_path:
        raise EnvironmentError("LOCAL_MODEL_PATH environment variable is not set.")
    with open(os.path.join(os.path.dirname(local_model_path), "info.json"), "w") as openfile:
        json.dump(info, openfile)


def CleanInfoJson():
    info_path = os.path.join(os.path.dirname(os.environ["LOCAL_MODEL_PATH"]), "info.json")
    if os.path.exists(info_path):
        os.remove(os.path.join(info_path))


def SetPaths(filename: str, items: Union[str, list]):
    if filename not in PathSet:
        raise ValueError("Unknown filename {}".format(filename))
    with open(os.path.join(os.environ["OUTPUT_PATH"], filename), "w") as F:
        if isinstance(items, str):
            F.write(items)
        else:
            F.write("\n".join(items))


def GetPaths(filename: str) -> list:
    if filename not in PathSet:
        raise ValueError("Unknown filename {}".format(filename))
    output_path = os.environ.get("OUTPUT_PATH")
    if not output_path:
        raise EnvironmentError("OUTPUT_PATH environment variable is not set.")
    with open(os.path.join(output_path, filename), "r") as F:
        content = F.read().splitlines()
    return content


def SaveGlobalInfoJson(infos: list, output_info_path: str):
    out = {"metadata": {}, "metrics": {}}

    for info in infos:

        with open(info, "r") as openfile:
            client_info_dict = json.load(openfile)

        # epoch
        out["metadata"]["epoch"] = client_info_dict["metadata"]["epoch"]

        # datasetSize
        if "datasetSize" not in out["metadata"]:
            out["metadata"]["datasetSize"] = [client_info_dict["metadata"]["datasetSize"]]
        else:
            out["metadata"]["datasetSize"].append(client_info_dict["metadata"]["datasetSize"])

        # metrics
        for metric in client_info_dict.get("metrics", {}):
            if metric not in out["metrics"]:
                out["metrics"][metric] = [client_info_dict["metrics"][metric]]
            else:
                out["metrics"][metric].append(client_info_dict["metrics"][metric])

    for metric in out["metrics"]:
        if "basic/" in metric:
            continue
        out["metrics"][metric] = sum(out["metrics"][metric]) / len(out["metrics"][metric])

    with open(output_info_path, "w") as openfile:
        json.dump(out, openfile)


def LoadGlobalInfoJson() -> dict:

    with open(
        os.path.join(os.path.dirname(os.environ["GLOBAL_MODEL_PATH"]), "merged-info.json"), "r"
    ) as openfile:
        merged_info = json.load(openfile)

    return merged_info


def compareVersion(v1: str, v2: str) -> int:
    v1, v2 = list(map(int, v1.split("."))), list(map(int, v2.split(".")))
    for rev1, rev2 in zip_longest(v1, v2, fillvalue=0):
        if rev1 == rev2:
            continue

        return -1 if rev1 < rev2 else 1

    return 0
