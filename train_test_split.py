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
    train_y, valid_y, test_y = [], [], []
    
    for n, val in enumerate(flood_freq):
        if val == "None":
            if val not in train:
                train[val] = [n]
                train_y.append(0)
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                    train_y.append(0)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(0)
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                            valid_y.append(0)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(0)
                            else:
                                test[val].append(n)
                                test_y.append(0)   
        elif val == "Rare":
            if val not in train:
                train[val] = [n]
                train_y.append(1)
            else:
                if len(train[val]) < split_idx[1]:
                    train[val].append(n)
                    train_y.append(1)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(1)
                    else:
                        if len(valid[val]) < int((freq_ratio[1] - split_idx[1]) / 2):
                            valid_y.append(1)
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test_y.append(1)
                                test[val] = [n]
                            else:
                                test_y.append(1)
                                test[val].append(n)  
        elif val == "Occasional":
            if val not in train:
                train[val] = [n]
                train_y.append(2)
            else:
                if len(train[val]) < split_idx[2]:
                    train_y.append(2)
                    train[val].append(n)
                else:
                    if val not in valid:
                        valid_y.append(2)
                        valid[val] = [n]
                    else:
                        if len(valid[val]) < int((freq_ratio[2] - split_idx[2]) / 2):
                            valid_y.append(2)
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test_y.append(2)
                                test[val] = [n]
                            else:
                                test_y.append(2)
                                test[val].append(n)  
        elif val == "Frequent":
            if val not in train:
                train[val] = [n]
                train_y.append(3)
            else:
                if len(train[val]) < split_idx[3]:
                    train[val].append(n)
                    train_y.append(3)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(3)
                    else:
                        if len(valid[val]) < int((freq_ratio[3] - split_idx[3]) / 2):
                            valid[val].append(n)
                            valid_y.append(3)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(3)
                            else:
                                test[val].append(n)  
                                test_y.append(3)
        elif val == "Very frequent":
            if val not in train:
                train[val] = [n]
                train_y.append(4)
            else:
                if len(train[val]) < split_idx[4]:
                    train[val].append(n)
                    train_y.append(4)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(4)
                    else:
                        if len(valid[val]) < int((freq_ratio[4] - split_idx[4]) / 2):
                            valid_y.append(4)
                            valid[val].append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(4)
                            else:
                                test[val].append(n)  
                                test_y.append(4)
        else:
            val = "None"
            if val not in train:
                train[val] = [n]
                train_y.append(0)
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                    train_y.append(0)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(0)
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                            valid_y.append(0)
                        else:
                            if val not in test:
                                test_y.append(0)
                                test[val] = [n]
                            else:
                                test_y.append(0)
                                test[val].append(n)  
    return train, valid, test, train_y, valid_y, test_y
