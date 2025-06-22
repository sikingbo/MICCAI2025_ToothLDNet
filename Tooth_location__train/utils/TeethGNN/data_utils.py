import numpy as np
import trimesh
from pathlib import Path
import json


def augment(vs, ts):
    # jitter vertices
    if np.random.rand(1) > 0.5:
        sigma, clip = 0.01, 0.05
        jitter = np.clip(sigma * np.random.randn(len(vs), 3), -1 * clip, clip)
        vs += jitter

    # translate
    if np.random.rand(1) > 0.5:
        scale = np.random.uniform(low=0.9, high=1.1, size=[3])
        translate = np.random.uniform(low=-0.2, high=0.2, size=[3])
        vs = np.add(np.multiply(vs, scale), translate)

    # change the order of vertices
    ts = np.roll(ts, np.random.randint(0, 3, 1), axis=1)

    # rotation
    if np.random.rand(1) > 0.5:
        axis_xyz = np.roll(np.eye(3), np.random.randint(0, 3, 1), axis=0)
        angles = np.random.uniform(low=-5/180*np.pi, high=5/180*np.pi, size=[3])
        matrix = trimesh.transformations.concatenate_matrices(*[trimesh.transformations.rotation_matrix(angle, axis) for axis, angle in zip(axis_xyz, angles)])
        vs = trimesh.transformations.transform_points(vs, matrix)

    return vs, ts


def load_predictions_json(fname: Path):

    cases = {}

    with open(fname, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {fname}")

    for e in entries:
        # Find case name through input file name
        inputs = e["inputs"]
        name = None
        for input in inputs:
            if input["interface"]["slug"] == "3d-teeth-scan":
                name = input["file"].split('/')[-1].split('.')[0] #str(input["image"]["name"])
                break  # expecting only a single input
        if name is None:
            raise ValueError(f"No filename found for entry: {e}")

        entry = {"name": name}

        # Find output value for this case
        outputs = e["outputs"]

        for output in outputs:
            if output["interface"]["slug"] == "dental-labels":
                # cases[name] = output['value']
                # cases[name] = e["pk"]
                cases[name] = output["file"]
    return cases


def get_offsets(points, labels):
    offsets = np.zeros((points.shape[0], 3), dtype='f4')
    for i in range(len(labels)):
        labels_t = labels[i]
        idx = np.where(labels_t != 0)[0]
        if len(idx) == 0:
            continue
        center = points[idx].mean(0)
        offset = center - points[idx, :3]
        offsets[idx] = offset
    return offsets

def test():
    mapping_dict = load_predictions_json(Path("D:/dataset/Teeth3DS/data/upper/00OMSZGW/00OMSZGW_upper.json"))
    return mapping_dict


if __name__ == "__main__":

    mapping_dict = test()
    print()

