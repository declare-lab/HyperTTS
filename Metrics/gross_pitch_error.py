import numpy as np
import torch
import torchaudio

from pitchTracking import Pitch

class GPE:
    def __init__(self, sr=16000):
        self.sr = sr
        self.pitch = Pitch(self.sr)
    
    def __call__(self, y_ref, y_syn):
        return self.calculate_gpe(y_ref, y_syn)

    def calculate_gpe_path(self, ref_path, syn_path):
        y_ref, sr_ref = torchaudio.load(ref_path)
        y_syn, sr_syn = torchaudio.load(syn_path)
        assert sr_ref == sr_syn, f"{sr_ref} != {sr_syn}" # audios of same sr
        assert sr_ref == self.sr, f"{sr_ref} != {self.sr}" # sr of audio and pitch tracking is same
        return self.calculate_gpe(y_ref, y_syn)

    def calculate_gpe(self, y_ref, y_syn):
        y_ref = y_ref.view(-1)
        y_syn = y_syn.view(-1)
        yref_f, _, _, yref_t = self.pitch.compute_yin(y_ref, self.sr)
        ysyn_f, _, _, ysyn_t = self.pitch.compute_yin(y_syn, self.sr)

        yref_f = np.array(yref_f)
        yref_t = np.array(yref_t)
        ysyn_f = np.array(ysyn_f)
        ysyn_t = np.array(ysyn_t)

        distortion = self.gross_pitch_error(yref_t, yref_f, ysyn_t, ysyn_f)
        return distortion.item()


    def gross_pitch_error(self, true_t, true_f, est_t, est_f):
        """The relative frequency in percent of pitch estimates that are
        outside a threshold around the true pitch. Only frames that are
        considered pitched by both the ground truth and the estimator (if
        applicable) are considered.
        """

        correct_frames = self._true_voiced_frames(true_t, true_f, est_t, est_f)
        gross_pitch_error_frames = self._gross_pitch_error_frames(
            true_t, true_f, est_t, est_f
        )
        return np.sum(gross_pitch_error_frames) / np.sum(correct_frames)

    def _true_voiced_frames(self, true_t, true_f, est_t, est_f):
        return (est_f != 0) & (true_f != 0)

    def _gross_pitch_error_frames(self, true_t, true_f, est_t, est_f, eps=1e-8):
        voiced_frames = self._true_voiced_frames(true_t, true_f, est_t, est_f)
        true_f_p_eps = [x + eps for x in true_f]
        pitch_error_frames = np.abs(est_f / true_f_p_eps - 1) > 0.2
        return voiced_frames & pitch_error_frames

if __name__ == "__main__":
    path1 = "../docs/audio/sp15.wav"
    path2 = "../docs/noisy/sp15_station_sn5.wav"

    gpe = GPE(22050)
    print(gpe.calculate_gpe_path(path1, path2))

    y_ref, sr_ref = torchaudio.load(path1)
    y_syn, sr_syn = torchaudio.load(path2)

    print(y_ref.size(), y_syn.size())
    print(gpe.calculate_gpe(y_ref, y_syn))
