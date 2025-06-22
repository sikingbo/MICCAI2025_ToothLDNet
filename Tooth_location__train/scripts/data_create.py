import trimesh
import json
import numpy as np
import os
import shutil
from mesh import TriMesh
from scripts.graph_cut import graph_cut


if __name__ == '__main__':
    # dir = ['lower', 'upper']
    obj_path = 'D:/dataset/Teeth3DS/data'
    box_txt = 'D:/dataset/Teeth_box/val.txt'
    output_path = 'D:/dataset/Teeth_box/data'
    json_file = 'D:/dataset/3DTeethLand/patches'
    i = 1
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            mesh_file = os.path.join(obj_path, dir, filename, filename + '_' + dir + '.obj')
            mesh_sim_file = os.path.join(obj_path, dir, filename, filename + '_' + dir + '_sim.off')
            # txt_file = os.path.join(obj_path, dir, filename, filename + '_' + dir + '_sim_box.txt')
            landmarks_file = os.path.join(json_file, filename + '_' + dir)
            if not (os.path.exists(mesh_file) and os.path.exists(mesh_sim_file)):
                continue

            output_file = os.path.join(output_path, filename + '_' + dir)
            if not os.path.exists(output_file):
                os.mkdir(output_file)
            output_obj = os.path.join(output_file, filename + '_' + dir + '.obj')
            output_sim_off = os.path.join(output_file, filename + '_' + dir + '_sim.off')

            shutil.copy2(mesh_file, output_obj)
            shutil.copy2(mesh_sim_file, output_sim_off)
            print(i, output_obj)
            i = i + 1

            # mesh = trimesh.load(output_obj)
            # labels = np.loadtxt(output_box, dtype=np.int32)
            # TriMesh(mesh.vertices, mesh.faces, labels).visualize()

            # os.remove(output_obj)

            patches_file = os.path.join(output_file, 'patches')
            if not os.path.exists(patches_file):
                os.mkdir(patches_file)
            for j in range(1, 17):
                patch_json_path = os.path.join(landmarks_file, str(j) + '.json')
                patch_npy_path = os.path.join(landmarks_file, str(j) + '.npy')
                patch_off_path = os.path.join(landmarks_file, str(j) + '.off')
                if not (os.path.exists(patch_json_path) and os.path.exists(patch_npy_path) and os.path.exists(patch_off_path)):
                    continue
                out_patch1 = os.path.join(patches_file, str(j) + '.json')
                out_patch2 = os.path.join(patches_file, str(j) + '.npy')
                out_patch3 = os.path.join(patches_file, str(j) + '.off')
                shutil.copy2(patch_json_path, out_patch1)
                shutil.copy2(patch_npy_path, out_patch2)
                shutil.copy2(patch_off_path, out_patch3)



