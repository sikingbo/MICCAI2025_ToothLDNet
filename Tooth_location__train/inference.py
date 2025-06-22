import torch, os, glob, click
import time
import numpy as np
import trimesh
from utils.predictor import Predictor
from utils.metrics import get_contour_points


@click.command()
@click.option('--data_path', default=R"D:/dataset/Teeth3DS/data/upper/RMZC48A0/RMZC48A0_upper_sim.off")
@click.option('--sample_num', default=8000, type=int, help='Number of points for the input of the network')
@click.option('--sample_times', default=10, type=int, help='Number of times for inference by the network, 0 indicating all faces will be sent into the network at least once')
@click.option('--weight_dir', type=str, default='D:\code/teeth\TeethLandCallenge\TeethGNN/runs/all_tooth/version_0/', help='')
def run(data_path, sample_num, sample_times, weight_dir):
    p = Predictor(weight_dir)
    m = trimesh.load_mesh(data_path)
    v_f = m.vertex_faces
    # c = get_contour_points(v_f, )
    t1 = time.time()
    labels = p.run(m, sample_num, sample_times)
    print(time.time()-t1)

    visual(m, labels)


def visual(m:trimesh.Trimesh, labels):
    from mesh import TriMesh
    TriMesh(m.vertices, m.faces, labels).visualize()


if __name__ == '__main__':
    run()
