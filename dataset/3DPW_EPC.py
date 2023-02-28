

class DHP19EPC(Dataset):
    def __init__(self, args, root_data_dir=None, root_label_dir=None,
                 root_3Dlabel_dir=None, root_dict_dir=None, min_EventNum=1024, Test3D=False):
        self.root_data_dir = root_data_dir
        self.root_label_dir = root_label_dir
        self.root_3Dlabel_dir = root_3Dlabel_dir
        self.Test3D = Test3D
        self.sample_point_num = args.num_points
        self.label = args.label
        self.sx = args.sensor_sizeW
        self.sy = args.sensor_sizeH