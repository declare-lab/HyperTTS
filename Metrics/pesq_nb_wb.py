import torchaudio
from pesq import pesq #install using pip install pesq

if __name__=="__main__":
    ## This implementation of PESQ supports both narrow band and wide band PESQ calculations
    path1 = "../docs/audio/sp15.wav"
    path2 = "../docs/noisy/sp15_station_sn5.wav"

    y_ref, sr = torchaudio.load(path1) #input the clean speech sample
    y_syn, sr = torchaudio.load(path2) # input the corresponding noisy speech sample

    y_ref = y_ref.numpy()[0]
    y_syn = y_syn.numpy()[0]
    score = pesq(sr, y_ref, y_syn, "nb")
    #score = pesq(sr, y_ref, y_syn, "wb") #no wide band mode if fs = 8000
    
    print(score)
