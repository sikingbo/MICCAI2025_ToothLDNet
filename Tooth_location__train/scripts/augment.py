import trimesh
import numpy as np
import random
import os




def random_translation(mesh, max_translation=0.1):
    """
    对网格应用随机平移变换
    :param mesh: Trimesh 3D 网格对象
    :param max_translation: 最大平移值（单位：与模型大小相关的比例）
    :return: 平移后的网格对象
    """
    augmented_mesh = mesh.copy()
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    augmented_mesh.apply_translation(translation)
    return augmented_mesh


def random_rotation(mesh):
    """
    对网格应用随机旋转变换
    :param mesh: Trimesh 3D 网格对象
    :return: 旋转后的网格对象
    """
    augmented_mesh = mesh.copy()
    # 随机生成旋转矩阵
    rotation_matrix = trimesh.transformations.random_rotation_matrix()
    augmented_mesh.apply_transform(rotation_matrix)
    return augmented_mesh


def random_scaling(mesh, scale_range=(0.8, 1.2)):
    """
    对网格应用随机缩放变换
    :param mesh: Trimesh 3D 网格对象
    :param scale_range: 缩放比例范围
    :return: 缩放后的网格对象
    """
    augmented_mesh = mesh.copy()
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    augmented_mesh.apply_scale(scale_factor)
    return augmented_mesh



if __name__ == '__main__':
    # dir = ['lower', 'upper']
    obj_path = 'E:/dataset/Teeth_box/data'
    box_txt = 'E:/dataset/Teeth_box/all.txt'
    i = 1
    with open(box_txt) as f:
        for line in f:
            filename = line.strip().split('_')[0]
            dir = line.strip().split('_')[1]  # ['lower', 'upper']
            tooth = filename + '_' + dir
            obj_file = os.path.join(obj_path, tooth, tooth + '_sim.off')
            output_files = os.path.join(obj_path, tooth, 'box_augment')
            if not os.path.exists(obj_file):
                continue
            print(i, obj_file)
            i = i + 1
            if not os.path.exists(output_files):
                os.mkdir(output_files)

            mesh = trimesh.load(obj_file)
            # 进行数据增强
            augmented_mesh = random_translation(mesh)
            augmented_mesh.export(os.path.join(output_files, 'translation.obj'))
            augmented_mesh = random_rotation(mesh)
            augmented_mesh.export(os.path.join(output_files, 'rotation.obj'))
            augmented_mesh = random_scaling(mesh)
            augmented_mesh.export(os.path.join(output_files, 'scaling.obj'))
