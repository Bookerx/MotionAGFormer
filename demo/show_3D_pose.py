import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def show3Dpose(vals, ax, show_label=True):
    ax.view_init(elev=15., azim=70)
    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)
    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in range(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    if show_label:
        for i in range(vals.shape[0]):
            ax.text(vals[i, 0], vals[i, 1], vals[i, 2], str(i), color='black', fontsize=9)

    RADIUS = 0.72
    RADIUS_Z = 0.7
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)

input_npz = "/home/xu/Desktop/MotionAGFormer/demo/output/debug/motionagformer-b-h36m_243/all_3d_keypoints.npz"
output_dir = "test"

data = np.load(input_npz, allow_pickle=True)
keypoints_list = data['reconstruction']

output_dir_3D = os.path.join(output_dir, 'pose3D/')
os.makedirs(output_dir_3D, exist_ok=True)
for idx, keypoints_dict in enumerate(keypoints_list):
    keypoints = np.array([
        keypoints_dict['root'], keypoints_dict['RHip'], keypoints_dict['RKnee'], keypoints_dict['RAnkle'],
        keypoints_dict['LHip'], keypoints_dict['LKnee'], keypoints_dict['LAnkle'], keypoints_dict['Spine'],
        keypoints_dict['Chest'], keypoints_dict['Neck'], keypoints_dict['Head'], keypoints_dict['LShoulder'],
        keypoints_dict['LElbow'], keypoints_dict['LWrist'], keypoints_dict['RShoulder'], keypoints_dict['RElbow'],
        keypoints_dict['RWrist']
    ])

    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')

    show3Dpose(keypoints, ax, show_label=True)
    plt.savefig(os.path.join(output_dir_3D, f'pose_{idx}.png'))
    plt.close(fig)
