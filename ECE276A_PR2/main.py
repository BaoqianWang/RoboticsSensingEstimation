import load_data as ld
import p2_utils as util

lidar0=ld.get_lidar("./lidar/train_lidar2")
ld.replay_lidar(lidar0)
joint = ld.get_joint("./joint/train_joint2")
