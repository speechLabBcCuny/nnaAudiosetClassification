

def randomMerge(data,count=1,splitIndex=-1):
    """
    Augment data by merging random samples. 
    Increase data size to augmentad_size.
    Select required number of samples from data and merge from given splitIndex
    assumes [batch,sequence,...]
    does not modify data
    """
    if count<1:
        return None
    
    sequenceLen=data.shape[1]
    sampleCount=data.shape[0]
    if splitIndex>=sequenceLen or splitIndex<-1:
        print("splitIndex should be less than sequence size")
        return None
    if sampleCount<1:
        print("needs at least 2 samples")
        return None
    
    # Augment post samples
    augmentad = data[:]
    NewOnes=np.empty((0,*augmentad.shape[1:]))
    patience=0
    while  NewOnes.shape[0] < count:
        left=count-NewOnes.shape[0]
        left= sampleCount if left>sampleCount else left
        new=np.empty((left,*augmentad.shape[1:]))
        #randomly select 2*left # of samples
        
        first = augmentad[torch.randperm(augmentad.shape[0])[:left]].reshape(-1,*augmentad.shape[1:])
        second = augmentad[torch.randperm(augmentad.shape[0])[:left]].reshape(-1,*augmentad.shape[1:])
        if splitIndex==-1:
            middle=torch.randint(0,sequenceLen,(1,))[0]
        else:
            middle=splitIndex
        new[:,0:middle],new[:,middle:]=first[:,0:middle],second[:,middle:]
        NewOnes=np.concatenate([NewOnes,new])
        NewOnes=np.unique(NewOnes,axis=0)
        patience+=1
        if patience>(count*10):
            print("samples not random enough")
            break
    return NewOnes




def randomConcat(data,y,count=1):
    """
    Augment data by concataneting random samples. 
    Generate count number of them.
    assumes [sampleIndex,sequence,...]
    does not modify data
    return array of new data and Y
    """
    if count<1:
        return None
    
    sequenceLen=data.shape[1]
    sampleCount=data.shape[0]

    if sampleCount<1:
        print("needs at least 2 samples")
        return None
    
    # Augment post samples
    augmentad = data[:]
    NewOnes=np.empty((0,augmentad.shape[1]*2,*augmentad.shape[2:]),dtype=augmentad.dtype)
    NewOnesY=np.empty((0,*y.shape[1:]),dtype=y.dtype)
    patience=0
    while  NewOnes.shape[0] < count:
        left=count-NewOnes.shape[0]
        left= sampleCount if left>sampleCount else left
#         new=np.empty((left,augmentad.shape[2]*2,*augmentad.shape[2:]))
        #randomly select 2*left # of samples
        firstIndexes = torch.randperm(augmentad.shape[0])[:left]
        secondIndexes = torch.randperm(augmentad.shape[0])[:left]
        first = augmentad[firstIndexes].reshape(-1,*augmentad.shape[1:])
        second = augmentad[secondIndexes].reshape(-1,*augmentad.shape[1:])
        firstY = y[firstIndexes]
        secondY = y[secondIndexes]
        new=np.concatenate([first,second],axis=1)
        newY = firstY | secondY
        
        NewOnes=np.concatenate([NewOnes,new])
        NewOnesY=np.concatenate([NewOnesY,newY])
        NewOnes,NewOnesIndex=np.unique(NewOnes,axis=0,return_index=True)
        NewOnesY=NewOnesY[NewOnesIndex]
        
        patience+=1
        if patience>(count*10):
            print("samples not random enough")
            break
    return NewOnes,NewOnesY


def randomAdd(data,y,count=1):
    """
    Augment data by adding random samples. 
    Generate count number of them.
    assumes [sampleIndex,sequence,...]
    does not modify data
    return array of new data and Y
    """
    if count<1:
        return None
    
    sequenceLen=data.shape[1]
    sampleCount=data.shape[0]

    if sampleCount<1:
        print("needs at least 2 samples")
        return None
    
    # Augment post samples
    augmentad = data[:]
    NewOnes=np.empty((0,augmentad.shape[1],*augmentad.shape[2:]),dtype=augmentad.dtype)
    NewOnesY=np.empty((0,*y.shape[1:]),dtype=y.dtype)
    patience=0
    while  NewOnes.shape[0] < count:
        left=count-NewOnes.shape[0]
        left= sampleCount if left>sampleCount else left
#         new=np.empty((left,augmentad.shape[2]*2,*augmentad.shape[2:]))
        #randomly select 2*left # of samples
        firstIndexes = torch.randperm(augmentad.shape[0])[:left]
        secondIndexes = torch.randperm(augmentad.shape[0])[:left]
        first = augmentad[firstIndexes].reshape(-1,*augmentad.shape[1:])
        second = augmentad[secondIndexes].reshape(-1,*augmentad.shape[1:])
        firstY = y[firstIndexes]
        secondY = y[secondIndexes]
        new = first*0.5 + second*0.5
#         new=np.concatenate([first,second],axis=1)
        newY = firstY | secondY
        
        NewOnes=np.concatenate([NewOnes,new])
        NewOnesY=np.concatenate([NewOnesY,newY])
        NewOnes,NewOnesIndex=np.unique(NewOnes,axis=0,return_index=True)
        NewOnesY=NewOnesY[NewOnesIndex]
        
        patience+=1
        if patience>(count*10):
            print("samples not random enough")
            break
    return NewOnes,NewOnesY