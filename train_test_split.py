def data_split(df):
    flood_freq = df["flodfreqdcd"]
    
    # [None, Rare, Occasional, Frequent, Very frequent]
    freq_ratio = [0] * 5
    
    for n, val in enumerate(flood_freq):
        if val == "None":
            freq_ratio[0] += 1
        elif val == "Rare":
            freq_ratio[1] += 1
        elif val == "Occasional":
            freq_ratio[2] += 1
        elif val == "Frequent":
            freq_ratio[3] += 1
        elif val == "Very frequent":
            freq_ratio[4] += 1
        else:
            freq_ratio[0] += 1
    
    split_idx = [int(x * 0.8) for x in freq_ratio]
    
    train, valid, test = dict(), dict(), dict()
    
    for n, val in enumerate(flood_freq):
        if val == "None":
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)   
        elif val == "Rare":
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[1]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[1] - split_idx[1]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)  
        elif val == "Occasional":
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[2]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[2] - split_idx[2]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)  
        elif val == "Frequent":
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[3]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[3] - split_idx[3]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)  
        elif val == "Very frequent":
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[4]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[4] - split_idx[4]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)  
        else:
            val = "None"
            if val not in train:
                train[val] = [n]
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                            else:
                                test[val].append(n)  
    return train, valid, test
