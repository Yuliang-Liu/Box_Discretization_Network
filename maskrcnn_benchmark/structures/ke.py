import torch


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class KES(object):
    def __init__(self, kes, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = kes.device if isinstance(kes, torch.Tensor) else torch.device('cpu')
        kes = torch.as_tensor(kes, dtype=torch.float32, device=device)
        if len(kes.size()) == 2:
            kes = kes.unsqueeze(2)
            if not kes.size()[0] ==0:
                assert(kes.size()[-2] == 12), str(kes.size()) # 12kes

        num_kes = kes.shape[0]
        kes_x = kes[:, :6, 0] # 4+2=6
        kes_y = kes[:, 6:, 0]
        # TODO remove once support or zero in dim is in
        if not kes.size()[0] ==0:
            assert(kes_x.size() == kes_y.size()), str(kes_x.size())+' '+str(kes_y.size())

        if num_kes > 0:
            kes = kes.view(num_kes, -1, 1)
            kes_x = kes_x.view(num_kes, -1, 1)
            kes_y = kes_y.view(num_kes, -1, 1)

        # TODO should I split them?
        self.kes = kes
        self.kes_x = kes_x
        self.kes_y = kes_y

        self.size = size
        self.mode = mode

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        k = self.kes.clone()
        k[:, :6, 0] -= box[0]
        k[:, 6:, 0] -= box[1]
        return type(self)(k, (w, h), self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data_x = self.kes_x.clone()
        resized_data_x[..., :] *= ratio_w

        resized_data_y = self.kes_y.clone()
        resized_data_y[..., :] *= ratio_h

        resized_data = torch.cat((resized_data_x, resized_data_y), dim=-2)
        return type(self)(resized_data, size, self.mode)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data_x = self.kes_x[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data_x[..., :] = width - flipped_data_x[..., :] - TO_REMOVE

        flipped_data_y = self.kes_y.clone()
        flipped_data = torch.cat((flipped_data_x, flipped_data_y), dim=-2)
        return type(self)(flipped_data, self.size, self.mode)

    def to(self, *args, **kwargs):
        return type(self)(self.kes.to(*args, **kwargs), self.size, self.mode)

    def __getitem__(self, item):
        return type(self)(self.kes[item], self.size, self.mode)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances_x={}, '.format(len(self.kes_x))
        s += 'num_instances_y={}, '.format(len(self.kes_y))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


class textKES(KES):
    NAMES = [ # x and y
        'meanx', 
        'xmin',
        'x2',
        'x3',
        'xmax',
        'cx'
        # 'meany',
        # 'ymin',
        # 'y2',
        # 'y3',
        # 'ymax',
        # 'cy'
    ]
    FLIP_MAP = {
        'xmin': 'xmax',
        'x2': 'x3',
    }


# TODO this doesn't look great
textKES.FLIP_INDS = _create_flip_indices(textKES.NAMES, textKES.FLIP_MAP)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def kes_to_heat_map(kes_x, kes_y, mty, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = kes_x[..., 0]
    y = kes_y[..., 0]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc_x = (x >= 0) & (x < heatmap_size)
    valid_x = (valid_loc_x).long()

    valid_loc_y = (y >= 0) & (y < heatmap_size)
    valid_y = (valid_loc_y).long()

    valid_mty = ((x >= 0) & (x < heatmap_size)) & ((y >= 0) & (y < heatmap_size))
    valid_mty = valid_mty.sum(dim=1)>0
    valid_mty = (valid_mty).long()

    heatmap_x = x
    heatmap_y = y

    mty = mty
    return heatmap_x, heatmap_y, valid_x, valid_y, mty, valid_mty
