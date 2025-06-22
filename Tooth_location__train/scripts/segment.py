import trimesh
import numpy as np


def segment_patch(mesh, labels):
    # lanels是点的标签
    vs = mesh.vertices
    fs = mesh.faces

    fv_labels = labels[fs]
    meshs = []
    maps = []
    for l in range(1, 17):
        v_idx = np.argwhere(labels == l).squeeze()
        if v_idx.shape[0] == 0:
            mesh_temp = trimesh.Trimesh()
            meshs.append(mesh_temp)
            maps.append([])
            continue
        vl = vs[v_idx]
        f_idx = np.where(np.all(fv_labels == l, axis=1))[0]
        fl = fs[f_idx]

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(v_idx)}
        new_faces = np.array([[index_mapping[vertex] for vertex in face] for face in fl])
        patch_mesh = trimesh.Trimesh(vertices=vl, faces=new_faces)
        # patch_mesh = trimesh.Trimesh(vertices=vs, faces=fl)
        meshs.append(patch_mesh)

        face_mapping = f_idx.tolist()  # 保存原始 mesh 中每个面的索引
        maps.append(f_idx)
    return meshs, maps

