import torch

def lm_score(model, samples, prefix_length):
   	# note to self, don't cut off in this case! 

def entropy(sen_lis, k):
	#sen_lis is like [['i','am','you','</s>'] ...]
	#assume it is lowered case, and clean
    dd, num = {}, 0
    for sen in sen_lis:
	   for i in range(0, len(sen) - k + 1):
		num += 1
		tt = ' '.join(sen[i:i+k])
		#print tt
		if not tt in dd: dd[tt] = 0
		dd[tt] += 1
    entro = 0.0
    for tt in dd:
	   prob = float(dd[tt] * 1.0) / num
	   entro = entro - math.log(prob) * prob
    return entro
