import trimesh
import numpy as np
from pygco import cut_from_graph
from mesh import TriMesh
from mesh import TriMesh


def graph_cut(faces, centers, normals, labels):
    num_classes = np.max(labels) + 1  # 假设类别从0开始
    one_hot_matrix = np.zeros((len(labels), num_classes))
    one_hot_matrix[np.arange(len(labels)), labels] = 1
    # refinement
    print('\tRefining by pygco...')
    round_factor = 100
    one_hot_matrix[one_hot_matrix < 1.0e-6] = 1.0e-6

    # unaries
    unaries = -round_factor * np.log10(one_hot_matrix)
    unaries = unaries.astype(np.int32)
    unaries = unaries.reshape(-1, num_classes)

    # parawise
    pairwise = (1 - np.eye(num_classes, dtype=np.int32))

    # edges
    lambda_c = 30
    edges = np.empty([1, 3], order='C')
    for i_node in range(faces.shape[0]):
        # Find neighbors
        nei = np.sum(np.isin(faces, faces[i_node, :]), axis=1)
        nei_id = np.where(nei == 2)
        for i_nei in nei_id[0][:]:
            if i_node < i_nei:
                cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]) / np.linalg.norm(
                    normals[i_node, 0:3]) / np.linalg.norm(normals[i_nei, 0:3])
                if cos_theta >= 1.0:
                    cos_theta = 0.9999
                theta = np.arccos(cos_theta)
                phi = np.linalg.norm(centers[i_node, :] - centers[i_nei, :])
                if theta > np.pi / 2.0:
                    edges = np.concatenate(
                        (edges, np.array([i_node, i_nei, -np.log10(theta / np.pi) * phi]).reshape(1, 3)), axis=0)
                else:
                    beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                    edges = np.concatenate(
                        (edges, np.array([i_node, i_nei, -beta * np.log10(theta / np.pi) * phi]).reshape(1, 3)), axis=0)
    edges = np.delete(edges, 0, 0)
    edges[:, 2] *= lambda_c * round_factor
    edges = edges.astype(np.int32)

    refine_labels = cut_from_graph(edges, unaries, pairwise)
    refine_labels = refine_labels.reshape([-1, 1])
    return refine_labels


def write_labels(label_file, write_file):
    # tar_labels = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
    tar_labels = [0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
    tar_labels = np.array(tar_labels)
    labels = np.loadtxt(label_file)
    for i, l in enumerate(labels):
        idx = np.argwhere(tar_labels == l)
        labels[i] = idx
    np.savetxt(write_file, labels, fmt='%d', delimiter=',')
    print(label_file)


import os
if __name__ == '__main__':
    # obj_path = 'E:/dataset/Teeth3DS/data/upper'
    # sub_files = os.listdir(obj_path)
    # for filename in sub_files:
    #     off_file = os.path.join(obj_path, filename, filename+'_upper_sim.off')
    #     re_file = os.path.join(obj_path, filename, filename + '_upper_sim_re.txt')
    #     # write_labels(off_file, re_file)
    #     labels = np.loadtxt(re_file, dtype='int64')
    #     mesh = trimesh.load_mesh(off_file)
    #     refine_labels = graph_cut(mesh.faces, mesh.triangles_center, mesh.face_normals, labels)
    #     np.savetxt(re_file, refine_labels, fmt='%d', delimiter=',')
    #     print(filename)

    off_file = 'D:/dataset/Teeth3DS/data/lower/0140W3ND/0140W3ND_lower_sim.off'
    txt_file = 'D:/dataset/Teeth3DS/data/lower/0140W3ND/0140W3ND_lower_box.txt'
    mesh = trimesh.load_mesh(off_file)
    labels = np.loadtxt(txt_file, dtype='int64')
    refine_labels = graph_cut(mesh.faces, mesh.triangles_center, mesh.face_normals, labels)
    # np.savetxt(txt_file, refine_labels, fmt='%d', delimiter=',')
    refine_labels = np.squeeze(refine_labels)
    TriMesh(mesh.vertices, mesh.faces, refine_labels).visualize()


