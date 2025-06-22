import json
import trimesh
import numpy as np
import matplotlib as mpl


def get_heatmap(off_file, json_file, dist_file):
    mesh = trimesh.load(off_file)
    v = np.array(mesh.vertices, dtype='float32')
    f = np.array(mesh.faces)
    landmarks = json.load(open(json_file))

    geodesic_matrix = np.load(dist_file)
    heatmaps = []
    cusp_heatmaps = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * 0.3 ** 2))
        # y = (dist - dist.min()) / (dist.max() - dist.min())
        # y = 1 - y
        heatmaps.append(y)
        if landmarks[i]['class'] == 'Cusp':
            cusp_heatmaps.append(y)
    cmap = mpl.colormaps['jet']
    cusp_heatmaps = np.array(cusp_heatmaps)
    cusp_heatmap = np.max(cusp_heatmaps, axis=0)

    for i, heatmap in enumerate(heatmaps):
        faces_colors = np.ones([f.shape[0], 4])
        for j, d in enumerate(heatmap):
            # d = (d+1)/2
            d = d * 2
            value = cmap(d)
            faces_colors[j][0] = value[0]
            faces_colors[j][1] = value[1]
            faces_colors[j][2] = value[2]
        mesh.visual.face_colors = faces_colors
        trimesh.exchange.export.export_mesh(mesh, 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/14_' + str(i) + '.obj', file_type='obj',  include_texture=True, include_color=True)

    # cusp shared
    faces_colors = np.ones([f.shape[0], 4])
    for j, d in enumerate(cusp_heatmap):
        # d = (d+1)/2
        d = d * 2
        value = cmap(d)
        faces_colors[j][0] = value[0]
        faces_colors[j][1] = value[1]
        faces_colors[j][2] = value[2]
    mesh.visual.face_colors = faces_colors
    trimesh.exchange.export.export_mesh(mesh, 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/14_cusp.obj', file_type='obj',
                                        include_texture=True, include_color=True)
    # trimesh.exchange.export.export_mesh(mesh, 'example_mesh_with_colors.obj', file_type='obj', include_texture=True, include_color=True)


def transform_off(off_file, write_file):
    mesh = trimesh.load(off_file)
    v = np.array(mesh.vertices, dtype='float32')
    f = np.array(mesh.faces)
    mesh.vertices = mesh.vertices - v.mean(0)
    trimesh.exchange.export.export_mesh(mesh, write_file, file_type='obj', include_texture=True,
                                        include_color=True)


def get_landmarks(off_file, json_file, dist_file):
    mesh = trimesh.load(off_file)
    cs = mesh.triangles_center
    landmarks = json.load(open(json_file))
    geo_dists = np.load(dist_file)
    coord_idx = np.argmin(geo_dists, axis=1)
    coords = cs[coord_idx]

    lm_idx = np.zeros(geo_dists.shape[0])
    for i, lm in enumerate(landmarks):
        if lm['class'] == 'Mesial':
            lm_idx[i] = 1
        if lm['class'] == 'Distal':
            lm_idx[i] = 2
        if lm['class'] == 'InnerPoint':
            lm_idx[i] = 3
        if lm['class'] == 'OuterPoint':
            lm_idx[i] = 4
        if lm['class'] == 'FacialPoint':
            lm_idx[i] = 5
        if lm['class'] == 'Cusp':
            lm_idx[i] = 6

    # pts = [trimesh.primitives.Sphere(radius=0.2, center=pt).to_mesh() for pt in coords]
    pts = []
    for i, center in enumerate(coords):
        if lm_idx[i] == 1:
            color = (255, 0, 0, 255)
        elif lm_idx[i] == 2:
            color = (0, 255, 0, 255)
        elif lm_idx[i] == 3:
            color = (255, 255, 0, 255)
        elif lm_idx[i] == 4:
            color = (0, 255, 255, 255)
        elif lm_idx[i] == 5:
            color = (255, 0, 255, 255)
        elif lm_idx[i] == 6:
            color = (0, 0, 255, 255)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.2, color=color)
        sphere.vertices += center
        pts.append(sphere)

    scene = trimesh.Scene(pts + [mesh])
    scene.export('C:/Users/Ricky/Desktop/teaser/01A6G_upper/14_landmarks.obj')
    # trimesh.exchange.export.export_mesh(mesh_out, 'C:/Users/Ricky/Desktop/teaser/13_lms.obj', file_type='obj', include_texture=True, include_color=True)


def get_jaw_landmarks(off_file, json_file):
    mesh = trimesh.load(off_file)
    cs = mesh.triangles_center
    annots = json.load(open(json_file))
    landmarks = annots['objects']
    coords = []
    lm_labels = []

    for dic in landmarks:
        if dic['class'] == 'Mesial':
            coords.append(dic['coord'])
            lm_labels.append(1)
        elif dic['class'] == 'Distal':
            coords.append(dic['coord'])
            lm_labels.append(2)
        elif dic['class'] == 'InnerPoint':
            coords.append(dic['coord'])
            lm_labels.append(3)
        elif dic['class'] == 'OuterPoint':
            coords.append(dic['coord'])
            lm_labels.append(4)
        elif dic['class'] == 'FacialPoint':
            coords.append(dic['coord'])
            lm_labels.append(5)
        else:
            coords.append(dic['coord'])
            lm_labels.append(6)

    # pts = [trimesh.primitives.Sphere(radius=0.2, center=pt).to_mesh() for pt in coords]
    pts = []
    for i, center in enumerate(coords):
        if lm_labels[i] == 1:
            color = (255, 0, 0, 255)
        elif lm_labels[i] == 2:
            color = (0, 255, 0, 255)
        elif lm_labels[i] == 3:
            color = (255, 255, 0, 255)
        elif lm_labels[i] == 4:
            color = (0, 255, 255, 255)
        elif lm_labels[i] == 5:
            color = (255, 0, 255, 255)
        elif lm_labels[i] == 6:
            color = (0, 0, 255, 255)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.5, color=color)
        sphere.vertices += center
        pts.append(sphere)

    scene = trimesh.Scene(pts + [mesh])
    scene.export('C:/Users/Ricky/Desktop/teaser/01A6G_upper/landmarks.obj')

import os
if __name__ == '__main__':
    # off_file = 'F:/dataset/3DTeethLand/patches/ZKJEPFDD_lower/13.off'
    off_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper/14.off'
    json_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper/14.json'
    dist_file = 'C:/Users/Ricky/Desktop/teaser/01A6G_upper/01A6GW4A_upper/14_f.npy'
    # write_file = 'C:/Users/Ricky/Desktop/teaser/13.obj'
    # transform_off(off_file, write_file)
    # get_heatmap(off_file, json_file, dist_file)
    get_landmarks(off_file, json_file, dist_file)
    # get_jaw_landmarks(off_file, json_file)

    # 创建一个球体网格
    # sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    # colors = np.random.randint(0, 255, size=(sphere.vertices.shape[0], 3))
    # sphere.visual.vertex_colors = colors
    # sphere.export('C:/Users/Ricky/Desktop/teaser/moved_sphere.ply')