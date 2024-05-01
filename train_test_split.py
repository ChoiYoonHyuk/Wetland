def data_split(df):
    wetland = df["wetland"]

    # [None, Rare, Occasional, Frequent, Very frequent]
    freq_ratio = [0] * 2
    
    for n, val in enumerate(wetland):
        if val == 0:
            freq_ratio[0] += 1
        else:
            freq_ratio[1] += 1
    
    #split_idx = [int(x * 0.8) for x in freq_ratio]
    split_idx = [int(freq_ratio[1] * 0.8), int(freq_ratio[1] * 0.8)]
    
    train, valid, test = dict(), dict(), dict()
    train_y, valid_y, test_y = [], [], []
    splits = [0, 0, 0]
    train_mask, valid_mask, test_mask, label, df_idx = [], [], [], [], []
    num_cls = [0] * 2
    
    wetland = df["wetland"]
    for n, val in enumerate(wetland):
        if val == 0:
            if len(train_y) < split_idx[1]:
                train_y.append(0)
                label.append(0)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                df_idx.append(n)
            else:
                if len(valid_y) < int((freq_ratio[1] - split_idx[1]) / 2):
                    valid_y.append(0)
                    label.append(0)
                    train_mask.append(False)
                    valid_mask.append(True)
                    test_mask.append(False)
                    df_idx.append(n)
                else:
                    if len(test_y) < int((freq_ratio[1] - split_idx[1]) / 2):
                        test_y.append(0)
                        label.append(0)
                        train_mask.append(False)
                        valid_mask.append(False)
                        test_mask.append(True)
                        df_idx.append(n)
        else:
            if splits[0] < split_idx[1]:
                splits[0] += 1
                label.append(1)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                df_idx.append(n)
            else:
                if splits[1] < int((freq_ratio[1] - split_idx[1]) / 2):
                    splits[1] += 1
                    label.append(1)
                    train_mask.append(False)
                    valid_mask.append(True)
                    test_mask.append(False)
                    df_idx.append(n)
                else:
                    if splits[2] < int((freq_ratio[1] - split_idx[1]) / 2):
                        splits[2] += 1
                        label.append(1)
                        train_mask.append(False)
                        valid_mask.append(False)
                        test_mask.append(True)
                        df_idx.append(n)

    #print(sum(train_mask), sum(valid_mask), sum(test_mask))
    return train_mask, valid_mask, test_mask, label, split_idx, df_idx
    
    
    
    '''flood_freq = df["Wetland"]
    
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
    freq_ratio[0] = 600
    split_idx[0] = 480
    
    train, valid, test = dict(), dict(), dict()
    train_y, valid_y, test_y = [], [], []
    train_mask, valid_mask, test_mask, label, df_idx = [], [], [], [], []
    num_cls = [0] * 5
    
    for n, val in enumerate(flood_freq):
        if val == "None":
            if val not in train:
                train[val] = [n]
                train_y.append(0)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(0)
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                    train_y.append(0)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(0)
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(0)
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(0)
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                            valid_y.append(0)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(0)
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(0)
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(0)
                                df_idx.append(n)
                                num_cls[0] += 1
                            else:
                                if len(test[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                                    test[val].append(n)
                                    test_y.append(0)  
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(0) 
                                    df_idx.append(n)
                                    num_cls[0] += 1
        elif val == "Rare":
            if val not in train:
                train[val] = [n]
                train_y.append(1)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(1) 
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[1]:
                    train[val].append(n)
                    train_y.append(1)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(1) 
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(1)
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(1) 
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[1] - split_idx[1]) / 2):
                            valid_y.append(1)
                            valid[val].append(n)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(1)
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test_y.append(1)
                                test[val] = [n]
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(1)
                                df_idx.append(n)
                                num_cls[1] += 1
                            else:
                                if len(test[val]) < int((freq_ratio[1] - split_idx[1]) / 2):
                                    test_y.append(1)
                                    test[val].append(n)
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(1)  
                                    df_idx.append(n)
                                    num_cls[1] += 1
        elif val == "Occasional":
            if val not in train:
                train[val] = [n]
                train_y.append(2)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(2) 
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[2]:
                    train_y.append(2)
                    train[val].append(n)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(2)
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid_y.append(2)
                        valid[val] = [n]
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(2)
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[2] - split_idx[2]) / 2):
                            valid_y.append(2)
                            valid[val].append(n)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(2)
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test_y.append(2)
                                test[val] = [n]
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(2)
                                df_idx.append(n)
                                num_cls[2] += 1
                            else:
                                if len(test[val]) < int((freq_ratio[2] - split_idx[2]) / 2):
                                    test_y.append(2)
                                    test[val].append(n)
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(2)  
                                    df_idx.append(n)
                                    num_cls[2] += 1
        elif val == "Frequent":
            if val not in train:
                train[val] = [n]
                train_y.append(3)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(3) 
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[3]:
                    train[val].append(n)
                    train_y.append(3)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(3) 
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(3)
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(3) 
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[3] - split_idx[3]) / 2):
                            valid[val].append(n)
                            valid_y.append(3)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(3) 
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(3)
                                num_cls[3] += 1
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(3) 
                                df_idx.append(n)
                            else:
                                if len(test[val]) < int((freq_ratio[3] - split_idx[3]) / 2):
                                    test[val].append(n)  
                                    test_y.append(3)
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(3) 
                                    df_idx.append(n)
                                    num_cls[3] += 1
        elif val == "Very frequent":
            if val not in train:
                train[val] = [n]
                train_y.append(4)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(4) 
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[4]:
                    train[val].append(n)
                    train_y.append(4)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(4) 
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(4)
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(4) 
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[4] - split_idx[4]) / 2):
                            valid_y.append(4)
                            valid[val].append(n)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(4) 
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(4)
                                num_cls[4] += 1
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(4) 
                                df_idx.append(n)
                            else:
                                if len(test[val]) < int((freq_ratio[4] - split_idx[4]) / 2):
                                    test[val].append(n)  
                                    test_y.append(4)
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(4) 
                                    df_idx.append(n)
                                    num_cls[4] += 1
        else:
            val = "None"
            if val not in train:
                train[val] = [n]
                train_y.append(0)
                train_mask.append(True)
                valid_mask.append(False)
                test_mask.append(False)
                label.append(0)
                df_idx.append(n)
            else:
                if len(train[val]) < split_idx[0]:
                    train[val].append(n)
                    train_y.append(0)
                    train_mask.append(True)
                    valid_mask.append(False)
                    test_mask.append(False)
                    label.append(0)
                    df_idx.append(n)
                else:
                    if val not in valid:
                        valid[val] = [n]
                        valid_y.append(0)
                        train_mask.append(False)
                        valid_mask.append(True)
                        test_mask.append(False)
                        label.append(0)
                        df_idx.append(n)
                    else:
                        if len(valid[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                            valid[val].append(n)
                            valid_y.append(0)
                            train_mask.append(False)
                            valid_mask.append(True)
                            test_mask.append(False)
                            label.append(0)
                            df_idx.append(n)
                        else:
                            if val not in test:
                                test[val] = [n]
                                test_y.append(0)
                                train_mask.append(False)
                                valid_mask.append(False)
                                test_mask.append(True)
                                label.append(0)
                                df_idx.append(n)
                                num_cls[0] += 1
                            else:
                                if len(test[val]) < int((freq_ratio[0] - split_idx[0]) / 2):
                                    test[val].append(n)
                                    test_y.append(0)  
                                    train_mask.append(False)
                                    valid_mask.append(False)
                                    test_mask.append(True)
                                    label.append(0) 
                                    df_idx.append(n)
                                    num_cls[0] += 1
    
    return train_mask, valid_mask, test_mask, label, split_idx, df_idx'''