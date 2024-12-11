import glob
import json
import os
from pathlib import Path

import pytest
import requests
from httpx import ASGITransport, AsyncClient


def download_file_from_google_drive(file_id, destination):
    # please refer to https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


@pytest.mark.asyncio
async def test_seg3d():
    seg3d_example = pytest.importorskip("test_tasks.seg3d_example")
    seg3d_app = seg3d_example.app

    # multiple dicom files
    src = "examples/inference/test_data/seg/img0062"
    input_files = glob.glob(src + "/*")
    if not input_files:
        print("Multiple dicom testing files are not found. Downloading...")
        download_file_from_google_drive("1E0ItBX2Ck654NGLuuDbbGHcPcbrPJgpg", src + ".zip")
        print("Multiple dicom download complete")
        os.makedirs(src, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(src + ".zip", "r") as zip_ref:
            zip_ref.extractall(src)
        os.remove(src + ".zip")

    files = []
    for filepath in input_files:
        filepath = Path(filepath)
        file = open(filepath, "rb")
        files.append(("files", (f"{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input_3d_dcm.json", "r") as f:
        data = json.load(f)
    for k in data:
        data[k] = json.dumps(data[k])

    with open("examples/inference/test_data/seg/input_3d_dcm_shuffle.json", "r") as f:
        data_shuffle = json.load(f)
    data_shuffle_drop = data_shuffle.copy()
    for k in data_shuffle:
        data_shuffle[k] = json.dumps(data_shuffle[k])
    for k in data_shuffle_drop:
        data_shuffle_drop[k] = json.dumps(data_shuffle_drop[k][10:])

    # single nifti file
    input_volumetric_files = "examples/inference/test_data/seg/img0062.nii.gz"
    if not os.path.exists(input_volumetric_files):
        print("Nifti testing file is not found. Downloading...")
        download_file_from_google_drive("17BoqW0oDn5Bos4iQLvBKiGJHt4fvhPVA", input_volumetric_files)
        print("Nifti file download complete")

    files_volume = []
    filepath = Path(input_volumetric_files)
    file = open(filepath, "rb")
    files_volume.append(("files", (f"{filepath.name}", file)))

    with open("examples/inference/test_data/seg/input_3d.json", "r") as f:
        data_volume = json.load(f)
    for k in data_volume:
        data_volume[k] = json.dumps(data_volume[k])

    async with AsyncClient(
        transport=ASGITransport(app=seg3d_app.app), base_url="http://test"
    ) as client:
        response = await client.post("/invocations", data=data, files=files)
        response_shuffle = await client.post("/invocations", data=data_shuffle, files=files)
        response_shuffle_drop = await client.post(
            "/invocations", data=data_shuffle_drop, files=files
        )
        response_volume = await client.post("/invocations", data=data_volume, files=files_volume)

        assert response.status_code == 200
        assert response_shuffle.status_code == 200
        assert response_shuffle_drop.status_code == 400  # files and images are not matched
        assert response_volume.status_code == 200

        ordered_content = json.loads(response.content)
        unordered_content = json.loads(response_shuffle.content)
        volumetric_content = json.loads(response_volume.content)

        for ordered, unordered, volumetric in zip(
            ordered_content["annotations"],
            unordered_content["annotations"],
            volumetric_content["annotations"],
        ):
            assert (
                ordered["segmentation"] == unordered["segmentation"] == volumetric["segmentation"]
            )

        for ordered, unordered, volumetric in zip(
            ordered_content["images"],
            unordered_content["images"],
            volumetric_content["images"],
        ):
            assert len(ordered) == len(unordered) == len(volumetric)
