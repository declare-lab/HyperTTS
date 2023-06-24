import torchaudio

from f0_frame_error import FFE
from gross_pitch_error import GPE
from mcd import MCD
from mcd2 import MCDMetric
from voicing_decision_error import VDE
from msd import MSD

from pystoi import stoi
from pesq import pesq

path1 = "../Testset/clean/sp15.wav"
path2 = "../Testset/noisy/station/sp15_station_sn5.wav"

y_ref, sr_ref = torchaudio.load(path1)
y_syn, sr_syn = torchaudio.load(path2) 

print("Clean and the noisy audios have lengths " + str(y_ref.size()[1]) + " and " + str(y_syn.size()[1]) + " respectively")

ffe = FFE(22050)
print("F0 Frame Error (FFE): " + str(ffe.calculate_ffe(y_ref, y_syn)))

gpe = GPE(22050)
print("Gross Pitch Error (GPE): " + str(gpe.calculate_gpe(y_ref, y_syn)))

mcd = MCD(22050)
print("MCD: " + str(mcd.calculate_mcd(y_ref, y_syn)))

mcd2 = MCDMetric(22050)
print("MCD2: " + str(mcd2(y_ref, y_syn)))

msd = MSD(22050)
print("MSD: "+ str(msd.calculate_msd(y_ref, y_syn)))

vde = VDE(22050)
print("Voice Decision Error (VDE): "+ str(vde(y_ref, y_syn)))

pesq_nb = pesq(sr_ref, y_ref.numpy()[0], y_syn.numpy()[0], "nb")
#pesq_wb = pesq(sr_ref, y_ref.numpy()[0], y_syn.numpy()[0], "wb")
print("Perceptual Evaluation of Speech Quality (PESQ)- Narrow Band: " + str(pesq_nb))
#print(pesq_wb)

stoi = stoi(y_ref.numpy()[0], y_syn.numpy()[0], sr_ref, extended=False)
print("Short Term Objective Intelligibility (STOI): " + str(stoi))