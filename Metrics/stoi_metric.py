import torchaudio
from pystoi import stoi

if __name__== "__main__":
    path1 = "../docs/audio/sp15.wav"
    path2 = "../docs/noisy/sp15_station_sn5.wav"
    y_ref, sr_ref = torchaudio.load(path1)
    y_ref, sr_syn = torchaudio.load(path2)
    stoi_score = stoi(y_ref.numpy()[0], y_syn.numpy()[0], sr_ref, extended=False)
    print(stoi_score)
