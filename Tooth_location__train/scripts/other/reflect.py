import os
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors


if __name__ == '__main__':
    teeth_path = 'D:/dataset/Teeth_box/data'
    box_txt = 'D:/dataset/Teeth_box/all.txt'
    i = 1
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            tooth_ori_file = os.path.join(teeth_path, tooth, tooth + '.obj')
            tooth_sim_file = os.path.join(teeth_path, tooth, tooth + '_sim.off')
            out_path = os.path.join(teeth_path, tooth, tooth + '_mapping.npy')
            out_v_path = os.path.join(teeth_path, tooth, tooth + '_v_mapping.npy')

            # mesh_ori = trimesh.load_mesh(tooth_ori_file)
            # mesh_sim = trimesh.load_mesh(tooth_sim_file)
            #
            # v_mesh_ori = mesh_ori.vertices
            # cs_mesh_sim = mesh_sim.triangles_center
            # cs_mesh_sim = cs_mesh_sim[:9998]

            # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cs_mesh_sim)
            # distances, indices = nbrs.kneighbors(v_mesh_ori)
            # indices = indices.astype(int).flatten()
            if os.path.exists(out_v_path):
                os.remove(out_v_path)
            # np.save(out_v_path, indices.flatten())
            print(f"{i}:Mapping saved to '{tooth}.npy'.")
            i = i + 1