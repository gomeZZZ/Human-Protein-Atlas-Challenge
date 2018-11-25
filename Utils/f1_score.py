def f1_score(labels,predictions,num_classes=28):
    """ Computes mean f1_score for a batch of labels and predictions in binary encoding [0,1,0...] for each class
    """
    f1 = 0
    # print(labels)
    # print(predictions)
    for i in range(labels.shape[0]):
        label = labels[i]
        pred = predictions[i]
        tp,fp,fn = 0,0,0
        
        for j in range(num_classes):
            if label[j] == 1 and pred[j] == 1: #True Positive
                tp+=1
            elif label[j] == 0 and pred[j] == 1: #False Positive
                fp+=1
            elif label[j] == 1 and pred[j] == 0: #False Negative
                fn+=1
            
        if tp+fp == 0: #if no positives, all were correct
            precision = 1
        else:
            precision = tp / (tp + fp) 
            
        if tp + fn == 0: #if no relevant cases, we hit them all
            recall = 1
        else:
            recall = tp / (tp+fn)
        
        if precision + recall == 0: #if we hit nothing...
            f1 += 0
        else:
            f1+= (2 * precision * recall) / (precision + recall)
    
    f1 = f1 / labels.shape[0]

    return f1