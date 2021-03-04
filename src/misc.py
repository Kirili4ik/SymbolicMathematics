import torch
import multiprocessing as mp


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
