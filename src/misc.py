import torch
import multiprocessing as mp

### tree pos enc ###

def generate_positions(root_paths, max_width, max_depth):
    """
    root_paths: List([ch_ids]) of size seq_len, ch_ids \in [0, 1, ..., max_width)
    returns: Tensor [len(root_paths), max_width * max_depth]
    """
    for i, path in enumerate(root_paths):
        # stack-like traverse
        if len(root_paths[i]) > max_depth:
            root_paths[i] = root_paths[i][-max_depth:]
        # pad
        root_paths[i] = root_paths[i][::-1] + [max_width] * (max_depth - len(root_paths[i]))

    # (seq_len, max_d)
    root_path_tensor = torch.LongTensor(root_paths)

    # (max_w + 1, max_w); 1 more is for pad = torch.zeros()
    onehots = torch.zeros((max_width + 1, max_width))
    for i in range(max_width):
        onehots[i, i] = 1.0

    # -> (seq_len*max_d, max_w)
    embeddings = torch.index_select(onehots, dim=0, index=root_path_tensor.view(-1))
    # -> (seq_len, max_d, max_w)
    embeddings = embeddings.view(root_path_tensor.shape + (embeddings.shape[-1],))
    # -> (seq_len, max_d*max_w)
    embeddings = embeddings.view((root_path_tensor.shape[0], -1))
    return embeddings


class TreePositionalEncodings(torch.nn.Module):
    def __init__(self, emb_size, width, depth):
        super(TreePositionalEncodings, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.depth = depth
        self.width = width
        self.d_tree_param = emb_size // depth // width
        self.d_pos = emb_size
        self.p = torch.nn.Parameter(torch.ones(self.d_tree_param, dtype=torch.float32), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)
        print(self.p)

    def build_weights(self):
        d_tree_param = self.d_tree_param
        tree_params = torch.tanh(self.p)

        # reshape p -> (max_d, max_w, d_tree_param)
        tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(self.depth, self.width, 1)

        # arange max_d --repeat>> (max_d, max_w, d_tree_param)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=self.p.device) \
            .reshape(-1, 1, 1).repeat(1, self.width, d_tree_param)
        # (d_tree_param)
        tree_norm = torch.sqrt((1 - tree_params ** 2) * self.d_pos / 2)

        # params ** depths * norm_coefs --reshape>> (max_d*max_w, d_tree_param)
        tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm) \
            .reshape(self.depth * self.width, d_tree_param)
        return tree_weights

    def treeify_positions(self, positions, tree_weights):
        # (bs, seq_len, max_w*max_d) * (max_d*max_w, d_tree_param) -> (bs, seq_len, max_w*max_d, d_tree_param)
        treeified = positions.unsqueeze(-1) * tree_weights

        # (bs, seq_len) + (emb_size,)  ->  (bs, seq_len, emb_size)
        shape = treeified.shape[:-2] + (self.d_pos,)
        return treeified.reshape(shape)

    def forward(self, positions):
        """
            positions: Tensor [bs, seq_len, max_w * max_d]
            returns: Tensor [bs, seq_len, max_w * max_d * n_features] = [bs, seq_len, emb_size]
        """
        tree_weights = self.build_weights()
        print('tree_weights are created of size', tree_weights.size())
        positions = self.treeify_positions(positions, tree_weights)
        print('result is positions', positions.size(), positions)
        return positions


### tree rel att ###

def generate_relative_positions_matrix(length,
                                       max_relative_positions,
                                       use_neg_dist,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)

    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    if use_neg_dist:
        final_mat = distance_mat_clipped + max_relative_positions
    else:
        final_mat = torch.abs(distance_mat_clipped)
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0]
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3)
    x_t_r = x_t.reshape(length, heads * batch_size, -1)
    if transpose:
        z_t = z.transpose(1, 2)
        x_tz_matmul = torch.matmul(x_t_r, z_t)
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1)
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t


def get_rel_mask(lengths, max_len):
    """
    Creates a boolean 3d mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    Used in tree relative attention
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    rel_matrix_mask = torch.zeros(batch_size, max_len, max_len, device=lengths.device)
    for i, l in enumerate(lengths):
        rel_matrix_mask[i, :l, :l] = 1
    return rel_matrix_mask

### parallel file readers ###

class FileReader:
    def __init__(self, filename):
        self.fin = open(filename, "r")
        self.line_map = list()             # Map from line index -> file position.
        self.line_map.append(0)
        i = 0
        while self.fin.readline():
            self.line_map.append(self.fin.tell())
            i += 1
            if i % 5_000_000 == 0:
                print('read', i)

    def get_line(self, index):
        self.fin.seek(self.line_map[index])
        return self.fin.readline()


class FileReaders:
    def __init__(self, filename, num_workers):
        self.filename = filename
        self.fds = []
        self.locks = []
        self.num_fds = 0
        self.global_lock = mp.Lock()
        #for i in range(num_workers):
        #    print('num worker number ', i)
        #    self.locks.append(False)
        #    self.fds.append(FileReader(self.filename))
        #    self.num_fds += 1

    def get_fd(self):
        res = -1
        with self.global_lock:
            for i in range(self.num_fds):
                if not self.locks[i]:
                    res = i
                    break
            if res == -1:
                self.locks.append(False)
                self.fds.append(FileReader(self.filename))
                res = self.num_fds
                self.num_fds += 1
            self.locks[res] = True
        return self.fds[res], res
