import numpy as np
import trimesh
import os


def file_select(folder, args):
    files = []
    if 'final' in folder:
        for tid in args.final_incisor_tids:
            mesh_file = os.path.join(folder, f'{tid}_1.stl')
            json_file = os.path.join(folder, f'{tid}_1.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((True, tid, mesh_file, json_file))
        for tid in args.final_canine_tids + args.final_premolar_tids:
            mesh_file = os.path.join(folder, f'{tid}_2.stl')
            json_file = os.path.join(folder, f'{tid}_2.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((True, tid, mesh_file, json_file))
        for tid in args.final_molar_tids:
            mesh_file = os.path.join(folder, f'{tid}_4.stl')
            json_file = os.path.join(folder, f'{tid}_4.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((True, tid, mesh_file, json_file))
    else:
        for tid in args.incisor_tids:
            mesh_file = os.path.join(folder, f'{tid}._Crown.stl')
            json_file = os.path.join(folder, f'{tid}._Crown.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((False, tid, mesh_file, json_file))
        for tid in args.canine_tids:
            mesh_file = os.path.join(folder, f'{tid}._Crown.stl')
            json_file = os.path.join(folder, f'{tid}._Crown.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((False, tid, mesh_file, json_file))
        for tid in args.premolar_tids:
            mesh_file = os.path.join(folder, f'{tid}._Crown.stl')
            json_file = os.path.join(folder, f'{tid}._Crown.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((False, tid, mesh_file, json_file))
        for tid in args.molar_tids:
            mesh_file = os.path.join(folder, f'{tid}._Crown.stl')
            json_file = os.path.join(folder, f'{tid}._Crown.json')
            if os.path.exists(mesh_file) and os.path.exists(json_file):
                files.append((False, tid, mesh_file, json_file))
    return files


def extract_landmark(landmarks, name):
    if name == 'outer_near_cusp':
        return landmarks['outer_cusp'][1]
    if name == 'outer_far_cusp':
        return landmarks['outer_cusp'][3]
    if name == 'inner_near_cusp':
        return landmarks['inner_cusp'][1]
    if name == 'inner_far_cusp':
        return landmarks['inner_cusp'][3]
    if 'fissure' in name:
        return landmarks['fissure'][int(name.split('_')[-1])]
    if name == 'fa_mean':
        return np.mean([landmarks[key][0] for key in ('fa', 'fa_shapely') if key in landmarks], axis=0)
    if 'front_cusp' in name:
        return landmarks['cusp'][int(name.split('_')[-1])]

    raise NotImplemented


def extract_final_landmark(landmarks):
    mid_curve = landmarks['MidCurve']
    facc = landmarks['Facc']
    feat_points = landmarks['FeatPoints']
    lms = np.concatenate([mid_curve, facc, feat_points], axis=0)
    return lms


def extract_gaussian_landmarks(tid, landmarks, centers, args):
    ys, masks = [], []
    if tid in args.incisor_tids:
        landmark_names = args.incisor_landmark_names
    elif tid in args.canine_tids:
        landmark_names = args.canine_landmark_names
    elif tid in args.premolar_tids:
        landmark_names = args.premolar_landmark_names
    else:
        landmark_names = args.molar_landmark_names
    for name in landmark_names:
        try:
            landmark = extract_landmark(landmarks, name)
            distance = np.linalg.norm(centers - landmark, axis=-1)
            y = np.exp(- distance ** 2 / (2 * args.landmark_std ** 2))
            masks.append(1)
        except (IndexError, KeyError):
            y = np.zeros((centers.shape[0], ))
            masks.append(0)
        ys.append(y)
    return np.asarray(ys).T, np.asarray(masks)


landmark_extractors = {
    'landmarks_gaussian': extract_gaussian_landmarks,
}


def get_heatmaps(pc, kp, args):
    ys = []
    for i, p in enumerate(kp):
        distance = np.linalg.norm(pc - p, axis=-1)
        y = np.exp(- distance ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


def geodesic_heatmaps(geodesic_matrix, args):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


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
    # if np.random.rand(1) > 0.5:
    #     axis_xyz = np.roll(np.eye(3), np.random.randint(0, 3, 1), axis=0)
    #     angles = np.random.uniform(low=-5/180*np.pi, high=5/180*np.pi, size=[3])
    #     matrix = trimesh.transformations.concatenate_matrices(*[trimesh.transformations.rotation_matrix(angle, axis) for axis, angle in zip(axis_xyz, angles)])
    #     vs = trimesh.transformations.transform_points(vs, matrix)

    return vs, ts


def augment_keypoint(vs):
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

    # rotation
    if np.random.rand(1) > 0.5:
        axis_xyz = np.roll(np.eye(3), np.random.randint(0, 3, 1), axis=0)
        angles = np.random.uniform(low=-5/180*np.pi, high=5/180*np.pi, size=[3])
        matrix = trimesh.transformations.concatenate_matrices(*[trimesh.transformations.rotation_matrix(angle, axis) for axis, angle in zip(axis_xyz, angles)])
        vs = trimesh.transformations.transform_points(vs, matrix)

    return vs


def show(vs, ts, y):
    from matplotlib import pyplot as plt

    mesh = trimesh.Trimesh(vs, ts)
    mesh.visual.face_colors = plt.get_cmap('jet')(y) * 255
    mesh.show()


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float64)
    colors = np.array(data[:, -1], dtype=np.int32)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


def vis_result(pc, g_kps, p_kps, write_file):
    note = open(write_file, mode='w')
    for i, vi in enumerate(pc):
        # if vi in p_kps:
        #     str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[0, 0,255] \n' % (vi[0], vi[1], vi[2])
        # else:
        str = 'sphere radius:0.005 pos:[%.10f,%.10f,%.10f] wirecolor:[169, 169,169] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)
    # for i, vi in enumerate(g_kps):
    #     str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[255, 0,0] \n' % (vi[0], vi[1], vi[2])
    #     note.writelines(str)
    for i, vi in enumerate(p_kps):
        str = 'sphere radius:0.02 pos:[%.10f,%.10f,%.10f] wirecolor:[0, 0,255] \n' % (vi[0], vi[1], vi[2])
        note.writelines(str)

    note.close()

