import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

all_types = [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],\
              [2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],[2,4,3,1],\
              [3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],\
              [4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
aty= [[all_types[iat][0]-1,all_types[iat][1]-1,all_types[iat][2]-1,all_types[iat][3]-1] for iat in range(24)]

class MTY(object):
    def __init__(self, mty, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = mty.device if isinstance(mty, torch.Tensor) else torch.device('cpu')
        mty = torch.as_tensor(mty, dtype=torch.int64, device=device)
            
        # TODO should I split them?
        assert(len(mty.size()) == 1), str(mty.size())
        self.mty = mty

        self.size = size
        self.mode = mode

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        return type(self)(self.mty, (w, h), self.mode)

    def resize(self, size, *args, **kwargs):
        return type(self)(self.mty, size, self.mode)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flipped_data = self.mty.clone()
        for i in range(self.mty.size()[0]):
            revs = [it for it in aty[self.mty[i]]]
            revs.reverse()
            flip_type = aty.index(revs)
            flipped_data[i] = flip_type

        return type(self)(flipped_data, self.size, self.mode)

    def to(self, *args, **kwargs):
        return type(self)(self.mty.to(*args, **kwargs), self.size, self.mode)

    def __getitem__(self, item):
        return type(self)(self.mty[item], self.size, self.mode)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.mty))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s
