import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

class MSD:
    def __init__(self, sr=16000):
        self.sr = sr

    def calculate_msd_path(self, ref_path, syn_path):
        y_ref, sr_ref = torchaudio.load(ref_path)
        y_syn, sr_syn = torchaudio.load(syn_path)
        assert sr_ref == sr_syn, f"{sr_ref} != {sr_syn}" # audios of same sr
        assert sr_ref == self.sr, f"{sr_ref} != {self.sr}" # sr of audio and pitch tracking is same
        return self.calculate_msd(y_ref, y_syn)

    def __call__(self, y_ref, y_syn):
        return self.calculate_msd(y_ref, y_syn)

    def calculate_msd(self, y_ref, y_syn, normalize_type="path"):
        # y_ref = y_ref.view(-1)
        # y_syn = y_syn.view(-1)

        melkwargs = {
            "n_fft": int(0.05 * self.sr), "win_length": int(0.05 * self.sr),
            "hop_length": int(0.0125 * self.sr), "f_min": 20,
            "n_mels": 80, "window_fn": torch.hann_window
        }
        
        mel_fn = torchaudio.transforms.MelSpectrogram(
            self.sr, n_fft=int(0.05 * self.sr), win_length=int(0.05 * self.sr),
            hop_length=int(0.0125 * self.sr), f_min=20, n_mels=80,
            window_fn=torch.hann_window
        )
        offset = 1e-6
        return self.batch_compute_distortion(y_ref, y_syn, self.sr, lambda y: torch.log(mel_fn(y) + offset).transpose(-1, -2), self.compute_rms_dist,normalize_type)

    def compute_l2_dist(self, x1, x2):
        """compute an (m, n) L2 distance matrix from (m, d) and (n, d) matrices"""
        return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0).pow(2)

    def compute_rms_dist(self, x1, x2):
        l2_dist = self.compute_l2_dist(x1, x2)
        return (l2_dist / x1.size(1)).pow(0.5)

    def antidiag_indices(self, offset, min_i=0, max_i=None, min_j=0, max_j=None):
        """
        for a (3, 4) matrix with min_i=1, max_i=3, min_j=1, max_j=4, outputs
        offset=2 (1, 1),
        offset=3 (2, 1), (1, 2)
        offset=4 (2, 2), (1, 3)
        offset=5 (2, 3)
        constraints:
            i + j = offset
            min_j <= j < max_j
            min_i <= offset - j < max_i
        """
        if max_i is None:
            max_i = offset + 1
        if max_j is None:
            max_j = offset + 1
        min_j = max(min_j, offset - max_i + 1, 0)
        max_j = min(max_j, offset - min_i + 1, offset + 1)
        j = torch.arange(min_j, max_j)
        i = offset - j
        return torch.stack([i, j])

    def get_divisor(self, pathmap, normalize_type):
        if normalize_type is None:
            return 1
        elif normalize_type == "len1":
            return pathmap.size(0)
        elif normalize_type == "len2":
            return pathmap.size(1)
        elif normalize_type == "path":
            return pathmap.sum().item()
        else:
            raise ValueError(f"normalize_type {normalize_type} not supported")

    def batch_dynamic_time_warping(self, distance, shapes=None):
        """full batched DTW without any constraints
        distance:  (batchsize, max_M, max_N) matrix
        shapes: (batchsize,) vector specifying (M, N) for each entry
        """
        # ptr: 0=left, 1=up-left, 2=up
        ptr2dij = {0: (0, -1), 1: (-1, -1), 2: (-1, 0)}

        bsz, m, n = distance.size()
        cumdist = torch.zeros_like(distance)
        backptr = torch.zeros_like(distance).type(torch.int32) - 1

        # initialize
        cumdist[:, 0, :] = distance[:, 0, :].cumsum(dim=-1)
        cumdist[:, :, 0] = distance[:, :, 0].cumsum(dim=-1)
        backptr[:, 0, :] = 0
        backptr[:, :, 0] = 2

        # DP with optimized anti-diagonal parallelization, O(M+N) steps
        for offset in range(2, m + n - 1):
            ind = self.antidiag_indices(offset, 1, m, 1, n)
            c = torch.stack(
                [cumdist[:, ind[0], ind[1] - 1], cumdist[:, ind[0] - 1, ind[1] - 1],
                cumdist[:, ind[0] - 1, ind[1]], ],
                dim=2
            )
            v, b = c.min(axis=-1)
            backptr[:, ind[0], ind[1]] = b.int()
            cumdist[:, ind[0], ind[1]] = v + distance[:, ind[0], ind[1]]

        # backtrace
        pathmap = torch.zeros_like(backptr)
        for b in range(bsz):
            i = m - 1 if shapes is None else (shapes[b][0] - 1).item()
            j = n - 1 if shapes is None else (shapes[b][1] - 1).item()
            dtwpath = [(i, j)]
            while (i != 0 or j != 0) and len(dtwpath) < 10000:
                assert (i >= 0 and j >= 0)
                di, dj = ptr2dij[backptr[b, i, j].item()]
                i, j = i + di, j + dj
                dtwpath.append((i, j))
            dtwpath = dtwpath[::-1]
            indices = torch.from_numpy(np.array(dtwpath))
            pathmap[b, indices[:, 0], indices[:, 1]] = 1

        return cumdist, backptr, pathmap

    def batch_compute_distortion(self, y1, y2, sr, feat_fn, dist_fn, normalize_type):
        d, s, x1, x2 = [], [], [], []
        for cur_y1, cur_y2 in zip(y1, y2):
            assert (cur_y1.ndim == 1 and cur_y2.ndim == 1)
            cur_x1 = feat_fn(cur_y1)
            cur_x2 = feat_fn(cur_y2)
            x1.append(cur_x1)
            x2.append(cur_x2)

            cur_d = dist_fn(cur_x1, cur_x2)
            d.append(cur_d)
            s.append(d[-1].size())
        max_m = max(ss[0] for ss in s)
        max_n = max(ss[1] for ss in s)
        d = torch.stack(
            [F.pad(dd, (0, max_n - dd.size(1), 0, max_m - dd.size(0))) for dd in d]
        )
        s = torch.LongTensor(s).to(d.device)
        cumdists, backptrs, pathmaps = self.batch_dynamic_time_warping(d, s)

        rets = []
        itr = zip(s, x1, x2, d, cumdists, backptrs, pathmaps)
        for (m, n), cur_x1, cur_x2, dist, cumdist, backptr, pathmap in itr:
            cumdist = cumdist[:m, :n]
            backptr = backptr[:m, :n]
            pathmap = pathmap[:m, :n]
            divisor = self.get_divisor(pathmap, normalize_type)

            distortion = cumdist[-1, -1] / divisor
            ret = distortion, (cur_x1, cur_x2, dist, cumdist, backptr, pathmap)
            rets.append(ret)

        return rets[0][0].item()
    

if __name__ == "__main__":
    path1 = "../docs/audio/sp15.wav"
    path2 = "../docs/audio/sp15_station_sn5.wav"

    msd = MSD(22050)

    y_ref, sr_ref = torchaudio.load(path1)
    y_syn, sr_syn = torchaudio.load(path2)

    print(msd.calculate_msd(y_ref, y_syn))
