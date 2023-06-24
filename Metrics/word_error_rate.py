import sys
import numpy as np

class WER:
    def __init__(self):
        self = self

    def levenshteinDistance(self, ref, hyp):
        '''
        This function calculates the Levenshtein Distance between the reference and the hypothesis sentence.
        '''
        ref_len = len(ref) + 1
        hyp_len = len(hyp) + 1
        dp = np.zeros(ref_len*hyp_len, dtype=np.uint8).reshape(ref_len, hyp_len)
        for i in range(ref_len):
            dp[i][0] = i
        for j in range(hyp_len):
            dp[0][j] = j
        for i in range(1, ref_len):
            for j in range(1, hyp_len):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    subs = dp[i-1][j-1] + 1
                    ins  = dp[i][j-1]   + 1
                    delt = dp[i-1][j]   + 1
                    dp[i][j] = min(subs, ins, delt)
        return dp

    def calculate_wer(self, ref, hyp):
        '''
        This calculates the word error rate given the reference and the hypothesis sentences.
        Returns the rate in percentage.
        Usage:
            >>>wer("Hi I love apples".split(), "Hi I love oranges".split())
            25.0
        '''
        # build the matrix for levenshteinDistance
        dp = self.levenshteinDistance(ref, hyp)
        rate = float(dp[len(ref)][len(hyp)]) / len(ref) * 100
        return rate

if __name__ == "__main__":
    ref = "Hi I love apples"
    hyp = "Hi I love oranges"
    wer = WER()
    print(wer.calculate_wer(ref.split(), hyp.split()))
