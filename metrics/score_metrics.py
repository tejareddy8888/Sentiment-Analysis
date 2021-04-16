def classDefiner(x): 
	if x[0] > x[1]:
		return -1
	return 1
	
def classDefiner_accuracy(x): 
    if x[0] > x[1]:
        return 0
    return 1

def accuracy(predicted,labels):
    acc = 0
    preds = map(classDefiner_accuracy,list(predicted.cpu().detach().numpy()))
    labels = list(labels.cpu().detach().numpy())
    preds = list(preds)
    for i in range(0,len(preds)):
      if preds[i] == labels[i]:
        acc+=1
    return acc/len(preds)
