from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from skimage import measure

def save_output(output_arr, output_size, output_dir, file_idx):
    plot_out_arr = np.array([])
    with_border_arr = np.zeros([34, 34, 34])
    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                plot_out_arr = np.append(plot_out_arr, output_arr[x_i, y_j, z_k])
                
    text_save = np.reshape(plot_out_arr, (output_size * output_size * output_size))
    np.savetxt(output_dir + '/volume' + str(file_idx) + '.txt', text_save)

    output_image = np.reshape(plot_out_arr, (output_size, output_size, output_size)).astype(np.float32)

    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                with_border_arr[x_i + 1, y_j + 1, z_k + 1] = output_image[x_i, y_j, z_k]

    if not np.any(with_border_arr):
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes_lewiner(with_border_arr, level = 0.0, gradient_direction = 'descent')
        faces = faces + 1

    obj_save = open(output_dir + '/volume' + str(file_idx) + '.obj', 'w')
    for item in verts:
        obj_save.write('v {0} {1} {2}\n'.format(item[0], item[1], item[2]))
    for item in normals:
        obj_save.write('vn {0} {1} {2}\n'.format(-item[0], -item[1], -item[2]))
    for item in faces:
        obj_save.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(item[0], item[2], item[1]))
    obj_save.close()

    output_image = np.rot90(output_image)
    x, y, z = output_image.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(x, y, z, zdir = 'z', c = 'red')
    plt.savefig(output_dir + '/volume' + str(file_idx) + '.png')
    plt.close()
