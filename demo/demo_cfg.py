"""
Run demo with different checkpoints.

checkpoint options:
    motionagformer-b-h36m.pth.tr        motionagformer-l-h36m.pth.tr
        frame_num = 243 or 81               frame_num = 243 or 81
        layer_num = 16                      layer_num = 26
        dimension_feat = 128                dimension_feat = 128

    motionagformer-s-h36m.pth.tr        motionagformer-xs-h36m.pth.tr
        frame_num =  81                     frame_num = 27
        layer_num = 26                      layer_num = 12
        dimension_feat = 64                dimension_feat = 64

"""

checkpoint = "../checkpoint/motionagformer-s-h36m.pth.tr"
frame_num = 81
layer_num = 26
dimension_feat = 64
generate_demo_video = True
