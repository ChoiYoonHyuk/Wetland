import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def data_process(src, trg, df_trg, s_name, t_name):
    # Remove some columns with natural languages
    df = src.drop(columns=['engdwbdcd', 'engsldcd'])
    
    # Add coordinate information
    #maps = pd.read_excel(r"coordinate_texas.xlsx", sheet_name="TX1")
    #maps = pd.read_excel(r"simp_coordinate.xlsx", sheet_name="TX")
    feat = 0
    
    # Vectorize some columns with discrete values
    if s_name == 'texas':
        feat = ['loam', 'clay', 'sand', 'fine', 'soil', 'complex', 'silt', 'coarse', 'outcrop', 'pit', 'flood', 'peat', '1', '2', '3', '4', '10', '25', '100']
    elif s_name == 'florida':
        feat = ['loamy', 'sand', 'fine', 'Bonifay', 'Albany', 'Ortega', '1', '2', '3', '4', '10', '25', '100']
    elif s_name == 'oregon':
        feat = ['loam', 'complex', 'sand', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
    elif s_name == 'louisiana':
        feat = ['association', 'soils', 'complex', 'muck', 'loam', 'sandy', 'pits', 'slit', '1', '2', '3', '4', '10', '25', '100']
    elif s_name == 'seattle':
        feat = ['pits', 'complex', 'loam', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
    elif s_name == 'arizona':
        feat = ['loamy', 'complex', 'association', 'loam', 'sandy', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
    
    feat2 = [str(x) for x in range(0, 101, 1)] 
    #feat.extend(feat2)
    #df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15']] = pd.DataFrame([[0] * 15], index=df.index)

    muname = list(set(df['muname']))
    
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        for w in word:
            if w in feat:
                idx = feat.index(w)
                df.at[i, idx] = 1
    
    pad = len(feat) - 7
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        ls = [0, 0]
        idx = 0    
        for w in word:
            if w in feat2:
                ls[idx] = feat2.index(w)
                idx += 1
            if idx > 1:
                break
        if ls[1] == 1:
            df.at[i, pad] = 1
        elif ls[1] == 2:
            df.at[i, pad+1] = 1
        elif ls[1] == 3:
            df.at[i, pad+2] = 1
        elif ls[1] == 4:
            df.at[i, pad+3] = 1
        elif ls[1] > 4 and ls[1] < 11:
            df.at[i, pad+4] = 1
        elif ls[1] > 10 and ls[1] < 26:
            df.at[i, pad+5] = 1
        elif ls[1] > 25:
            df.at[i, pad+6] = 1
    df = df.drop(columns=['muname'])
    
    df = df.fillna(0)
    
    df['flodfreqdcd'] = df['flodfreqdcd'].fillna(0)
    flod = list(set(df['flodfreqdcd']))
    for i in range(len(flod)):
        df['flodfreqdcd'] = df['flodfreqdcd'].replace(flod[i], i)
    df['flodfreqdcd'] = df['flodfreqdcd'] / len(flod)
    
    df['drclassdcd'] = df['drclassdcd'].fillna(0)
    drcls = list(set(df['drclassdcd']))
    for i in range(len(drcls)):
        df['drclassdcd'] = df['drclassdcd'].replace(drcls[i], i)
    df['drclassdcd'] = df['drclassdcd'] / len(drcls)
    
    df['hydgrpdcd'] = df['hydgrpdcd'].fillna(0)
    hydgr = list(set(df['hydgrpdcd']))
    for i in range(len(hydgr)):
        df['hydgrpdcd'] = df['hydgrpdcd'].replace(hydgr[i], i)
    df['hydgrpdcd'] = df['hydgrpdcd'] / len(hydgr)
    
    df['engdwobdcd'] = df['engdwobdcd'].fillna(0)    
    engdw = list(set(df['engdwobdcd']))
    for i in range(len(engdw)):
        df['engdwobdcd'] = df['engdwobdcd'].replace(engdw[i], i)
    df['engdwobdcd'] = df['engdwobdcd'] / len(engdw)
        
    df['engdwbll'] = df['engdwbll'].fillna(0)    
    engdwbll = list(set(df['engdwbll']))
    for i in range(len(engdwbll)):
        df['engdwbll'] = df['engdwbll'].replace(engdwbll[i], i)
    df['engdwbll'] = df['engdwbll'] / len(engdwbll)
    
    df['engdwbml'] = df['engdwbml'].fillna(0)    
    engdwbml = list(set(df['engdwbml']))
    for i in range(len(engdwbml)):
        df['engdwbml'] = df['engdwbml'].replace(engdwbml[i], i)
    df['engdwbml'] = df['engdwbml'] / len(engdwbml)
    
    df['engstafdcd'] = df['engstafdcd'].fillna(0)    
    engstafdcd = list(set(df['engstafdcd']))
    for i in range(len(engstafdcd)):
        df['engstafdcd'] = df['engstafdcd'].replace(engstafdcd[i], i)
    df['engstafdcd'] = df['engstafdcd'] / len(engstafdcd)
    
    df['engstafll'] = df['engstafll'].fillna(0)    
    engstafll = list(set(df['engstafll']))
    for i in range(len(engstafll)):
        df['engstafll'] = df['engstafll'].replace(engstafll[i], i)
    df['engstafll'] = df['engstafll'] / len(engstafll)
    
    df['engstafml'] = df['engstafml'].fillna(0)    
    engstafml = list(set(df['engstafml']))
    for i in range(len(engstafml)):
        df['engstafml'] = df['engstafml'].replace(engstafml[i], i)
    df['engstafml'] = df['engstafml'] / len(engstafml)
    
    df['engsldcp'] = df['engsldcp'].fillna(0)    
    engsldcp = list(set(df['engsldcp']))
    for i in range(len(engsldcp)):
        df['engsldcp'] = df['engsldcp'].replace(engsldcp[i], i)
    df['engsldcp'] = df['engsldcp'] / len(engsldcp)
    
    df['englrsdcd'] = df['englrsdcd'].fillna(0)    
    englrsdcd = list(set(df['englrsdcd']))
    for i in range(len(englrsdcd)):
        df['englrsdcd'] = df['englrsdcd'].replace(englrsdcd[i], i)
    df['englrsdcd'] = df['englrsdcd'] / len(englrsdcd)
    
    df['engcmssdcd'] = df['engcmssdcd'].fillna(0)    
    engcmssdcd = list(set(df['engcmssdcd']))
    for i in range(len(engcmssdcd)):
        df['engcmssdcd'] = df['engcmssdcd'].replace(engcmssdcd[i], i)
    df['engcmssdcd'] = df['engcmssdcd'] / len(engcmssdcd)
    
    df['engcmssmp'] = df['engcmssmp'].fillna(0)    
    engcmssmp = list(set(df['engcmssmp']))
    for i in range(len(engcmssmp)):
        df['engcmssmp'] = df['engcmssmp'].replace(engcmssmp[i], i)
    df['engcmssmp'] = df['engcmssmp'] / len(engcmssmp)
    
    df['urbrecptdcd'] = df['urbrecptdcd'].fillna(0)    
    urbrecptdcd = list(set(df['urbrecptdcd']))
    for i in range(len(urbrecptdcd)):
        df['urbrecptdcd'] = df['urbrecptdcd'].replace(urbrecptdcd[i], i)
    df['urbrecptdcd'] = df['urbrecptdcd'] / len(urbrecptdcd)
    
    df['forpehrtdcp'] = df['forpehrtdcp'].fillna(0)    
    forpehrtdcp = list(set(df['forpehrtdcp']))
    for i in range(len(forpehrtdcp)):
        df['forpehrtdcp'] = df['forpehrtdcp'].replace(forpehrtdcp[i], i)
    df['forpehrtdcp'] = df['forpehrtdcp'] / len(forpehrtdcp)
    
    # Normalize some columns with continuous values
    df['slopegraddcp'] = df['slopegraddcp'].fillna(0)
    val = list(set(df['slopegraddcp']))
    df['slopegraddcp'] = df['slopegraddcp'] / max(val)
    
    df['slopegradwta'] = df['slopegradwta'].fillna(0)
    val = list(set(df['slopegradwta']))
    df['slopegradwta'] = df['slopegradwta'] / max(val)
    
    df['wtdepannmin'] = df['wtdepannmin'].fillna(0)
    val = list(set(df['wtdepannmin']))
    df['wtdepannmin'] = df['wtdepannmin'] / max(val)
    
    '''df['pondfreqprs'] = df['pondfreqprs'].fillna(0)
    val = list(set(df['pondfreqprs']))
    df['pondfreqprs'] = df['pondfreqprs'] / max(val)'''
    
    df['aws025wta'] = df['aws025wta'].fillna(0)
    val = list(set(df['aws025wta']))
    df['aws025wta'] = df['aws025wta'] / max(val)
    
    df['aws050wta'] = df['aws050wta'].fillna(0)
    val = list(set(df['aws050wta']))
    df['aws050wta'] = df['aws050wta'] / max(val)
    
    df['aws0100wta'] = df['aws0100wta'].fillna(0)
    val = list(set(df['aws0100wta']))
    df['aws0100wta'] = df['aws0100wta'] / max(val)
    
    df['aws0150wta'] = df['aws0150wta'].fillna(0)
    val = list(set(df['aws0150wta']))
    df['aws0150wta'] = df['aws0150wta'] / max(val)
    
    df['iccdcd'] = df['iccdcd'].fillna(0)
    val = list(set(df['iccdcd']))
    df['iccdcd'] = df['iccdcd'] / max(val)
    
    df['iccdcdpct'] = df['iccdcdpct'].fillna(0)
    val = list(set(df['iccdcdpct']))
    df['iccdcdpct'] = df['iccdcdpct'] / max(val)
    
    df['niccdcd'] = df['niccdcd'].fillna(0)
    val = list(set(df['niccdcd']))
    df['niccdcd'] = df['niccdcd'] / max(val)
    
    df['niccdcdpct'] = df['niccdcdpct'].fillna(0)
    val = list(set(df['niccdcdpct']))
    df['niccdcdpct'] = df['niccdcdpct'] / max(val)
    
    df['urbrecptwta'] = df['urbrecptwta'].fillna(0)
    val = list(set(df['urbrecptwta']))
    df['urbrecptwta'] = df['urbrecptwta'] / max(val)
    
    df['hydclprs'] = df['hydclprs'].fillna(0)
    val = list(set(df['hydclprs']))
    df['hydclprs'] = df['hydclprs'] / max(val)
    
    df['awmmfpwwta'] = df['awmmfpwwta'].fillna(0)
    val = list(set(df['awmmfpwwta']))
    df['awmmfpwwta'] = df['awmmfpwwta'] / max(val)
    
    # Construct adjacency matrix
    df_idx = []
    
    wetland = df["wetland"]
    for n, val in enumerate(wetland):
        df_idx.append(n)
                        
    df_idx.sort()
    
    df = df.iloc[df_idx]
    
    topk = 3
    x, y = df['x'], df['y']
    
    tmp = []
    for i in range(len(x)):
        tmp.append([x.iloc[i], y.iloc[i]])
    
    tmp = torch.tensor(tmp)
    cdist = torch.cdist(tmp, tmp).to(device)
    src_adj = torch.zeros(len(cdist), len(cdist)).to(device)
    for idx in range(len(cdist)):
        row = torch.topk(-cdist[idx], topk+1)
        src_adj[idx] = torch.where(-cdist[idx] >= -cdist[idx][row[1][topk]], 1, 0)
    
    df = df.fillna(0)
    
    src_label = df['wetland']
    df = df.drop(columns=['wetland'])
    src = df.drop(columns=['x', 'y'])
    
###################################################################################
###################################################################################
###################################################################################
    
    # Remove some columns with natural languages
    df = trg.drop(columns=['engdwbdcd', 'engsldcd'])
    
    # Add coordinate information
    #maps = pd.read_excel(r"coordinate_texas.xlsx", sheet_name="TX1")
    #maps = pd.read_excel(r"simp_coordinate.xlsx", sheet_name="TX")
    feat = 0
    
    # Vectorize some columns with discrete values
    if t_name == 'texas':
        feat = ['loam', 'clay', 'sand', 'fine', 'soil', 'complex', 'silt', 'coarse', 'outcrop', 'pit', 'flood', 'peat', '1', '2', '3', '4', '10', '25', '100']
    elif t_name == 'florida':
        feat = ['loamy', 'sand', 'fine', 'Bonifay', 'Albany', 'Ortega', '1', '2', '3', '4', '10', '25', '100']
    elif t_name == 'oregon':
        feat = ['loam', 'complex', 'sand', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
    elif t_name == 'louisiana':
        feat = ['association', 'soils', 'complex', 'muck', 'loam', 'sandy', 'pits', 'slit', '1', '2', '3', '4', '10', '25', '100']
    elif t_name == 'seattle':
        feat = ['pits', 'complex', 'loam', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
    elif t_name == 'arizona':
        feat = ['loam', 'complex', 'sand', 'peat', 'water', '1', '2', '3', '4', '10', '25', '100']
        
    feat2 = [str(x) for x in range(0, 101, 1)] 
    #feat.extend(feat2)
    #df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15']] = pd.DataFrame([[0] * 15], index=df.index)

    muname = list(set(df['muname']))
    
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        for w in word:
            if w in feat:
                idx = feat.index(w)
                df.at[i, idx] = 1
    
    pad = len(feat) - 7
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        ls = [0, 0]
        idx = 0    
        for w in word:
            if w in feat2:
                ls[idx] = feat2.index(w)
                idx += 1
            if idx > 1:
                break
        if ls[1] == 1:
            df.at[i, pad] = 1
        elif ls[1] == 2:
            df.at[i, pad+1] = 1
        elif ls[1] == 3:
            df.at[i, pad+2] = 1
        elif ls[1] == 4:
            df.at[i, pad+3] = 1
        elif ls[1] > 4 and ls[1] < 11:
            df.at[i, pad+4] = 1
        elif ls[1] > 10 and ls[1] < 26:
            df.at[i, pad+5] = 1
        elif ls[1] > 25:
            df.at[i, pad+6] = 1
    '''histo = [0] * 100
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        ls = [0, 0]
        idx = 0
        for w in word:
            if w in feat2:
                ls[idx] = feat2.index(w)
                idx += 1
        idx = 0
        tmp = [0] * 100
        if ls[0] != ls[1]:
            tmp[ls[0]:ls[1]] = [1] * (ls[1] - ls[0])
        histo = np.add(histo, tmp)
    print(histo)
    exit()'''
    df = df.drop(columns=['muname'])
    
    #for i in range(len(feat)):
    #    df[i] = df[i].fillna(0)
    df = df.fillna(0)
    
    df['flodfreqdcd'] = df['flodfreqdcd'].fillna(0)
    flod = list(set(df['flodfreqdcd']))
    for i in range(len(flod)):
        df['flodfreqdcd'] = df['flodfreqdcd'].replace(flod[i], i)
    df['flodfreqdcd'] = df['flodfreqdcd'] / len(flod)
    
    df['drclassdcd'] = df['drclassdcd'].fillna(0)
    drcls = list(set(df['drclassdcd']))
    for i in range(len(drcls)):
        df['drclassdcd'] = df['drclassdcd'].replace(drcls[i], i)
    df['drclassdcd'] = df['drclassdcd'] / len(drcls)
    
    df['hydgrpdcd'] = df['hydgrpdcd'].fillna(0)
    hydgr = list(set(df['hydgrpdcd']))
    for i in range(len(hydgr)):
        df['hydgrpdcd'] = df['hydgrpdcd'].replace(hydgr[i], i)
    df['hydgrpdcd'] = df['hydgrpdcd'] / len(hydgr)
    
    df['engdwobdcd'] = df['engdwobdcd'].fillna(0)    
    engdw = list(set(df['engdwobdcd']))
    for i in range(len(engdw)):
        df['engdwobdcd'] = df['engdwobdcd'].replace(engdw[i], i)
    df['engdwobdcd'] = df['engdwobdcd'] / len(engdw)
        
    df['engdwbll'] = df['engdwbll'].fillna(0)    
    engdwbll = list(set(df['engdwbll']))
    for i in range(len(engdwbll)):
        df['engdwbll'] = df['engdwbll'].replace(engdwbll[i], i)
    df['engdwbll'] = df['engdwbll'] / len(engdwbll)
    
    df['engdwbml'] = df['engdwbml'].fillna(0)    
    engdwbml = list(set(df['engdwbml']))
    for i in range(len(engdwbml)):
        df['engdwbml'] = df['engdwbml'].replace(engdwbml[i], i)
    df['engdwbml'] = df['engdwbml'] / len(engdwbml)
    
    df['engstafdcd'] = df['engstafdcd'].fillna(0)    
    engstafdcd = list(set(df['engstafdcd']))
    for i in range(len(engstafdcd)):
        df['engstafdcd'] = df['engstafdcd'].replace(engstafdcd[i], i)
    df['engstafdcd'] = df['engstafdcd'] / len(engstafdcd)
    
    df['engstafll'] = df['engstafll'].fillna(0)    
    engstafll = list(set(df['engstafll']))
    for i in range(len(engstafll)):
        df['engstafll'] = df['engstafll'].replace(engstafll[i], i)
    df['engstafll'] = df['engstafll'] / len(engstafll)
    
    df['engstafml'] = df['engstafml'].fillna(0)    
    engstafml = list(set(df['engstafml']))
    for i in range(len(engstafml)):
        df['engstafml'] = df['engstafml'].replace(engstafml[i], i)
    df['engstafml'] = df['engstafml'] / len(engstafml)
    
    df['engsldcp'] = df['engsldcp'].fillna(0)    
    engsldcp = list(set(df['engsldcp']))
    for i in range(len(engsldcp)):
        df['engsldcp'] = df['engsldcp'].replace(engsldcp[i], i)
    df['engsldcp'] = df['engsldcp'] / len(engsldcp)
    
    df['englrsdcd'] = df['englrsdcd'].fillna(0)    
    englrsdcd = list(set(df['englrsdcd']))
    for i in range(len(englrsdcd)):
        df['englrsdcd'] = df['englrsdcd'].replace(englrsdcd[i], i)
    df['englrsdcd'] = df['englrsdcd'] / len(englrsdcd)
    
    df['engcmssdcd'] = df['engcmssdcd'].fillna(0)    
    engcmssdcd = list(set(df['engcmssdcd']))
    for i in range(len(engcmssdcd)):
        df['engcmssdcd'] = df['engcmssdcd'].replace(engcmssdcd[i], i)
    df['engcmssdcd'] = df['engcmssdcd'] / len(engcmssdcd)
    
    df['engcmssmp'] = df['engcmssmp'].fillna(0)    
    engcmssmp = list(set(df['engcmssmp']))
    for i in range(len(engcmssmp)):
        df['engcmssmp'] = df['engcmssmp'].replace(engcmssmp[i], i)
    df['engcmssmp'] = df['engcmssmp'] / len(engcmssmp)
    
    df['urbrecptdcd'] = df['urbrecptdcd'].fillna(0)    
    urbrecptdcd = list(set(df['urbrecptdcd']))
    for i in range(len(urbrecptdcd)):
        df['urbrecptdcd'] = df['urbrecptdcd'].replace(urbrecptdcd[i], i)
    df['urbrecptdcd'] = df['urbrecptdcd'] / len(urbrecptdcd)
    
    df['forpehrtdcp'] = df['forpehrtdcp'].fillna(0)    
    forpehrtdcp = list(set(df['forpehrtdcp']))
    for i in range(len(forpehrtdcp)):
        df['forpehrtdcp'] = df['forpehrtdcp'].replace(forpehrtdcp[i], i)
    df['forpehrtdcp'] = df['forpehrtdcp'] / len(forpehrtdcp)
    
    # Normalize some columns with continuous values
    df['slopegraddcp'] = df['slopegraddcp'].fillna(0)
    val = list(set(df['slopegraddcp']))
    df['slopegraddcp'] = df['slopegraddcp'] / max(val)
    
    df['slopegradwta'] = df['slopegradwta'].fillna(0)
    val = list(set(df['slopegradwta']))
    df['slopegradwta'] = df['slopegradwta'] / max(val)
    
    df['wtdepannmin'] = df['wtdepannmin'].fillna(0)
    val = list(set(df['wtdepannmin']))
    df['wtdepannmin'] = df['wtdepannmin'] / max(val)
    
    df['pondfreqprs'] = df['pondfreqprs'].fillna(0)
    val = list(set(df['pondfreqprs']))
    df['pondfreqprs'] = df['pondfreqprs'] / max(val)
    
    df['aws025wta'] = df['aws025wta'].fillna(0)
    val = list(set(df['aws025wta']))
    df['aws025wta'] = df['aws025wta'] / max(val)
    
    df['aws050wta'] = df['aws050wta'].fillna(0)
    val = list(set(df['aws050wta']))
    df['aws050wta'] = df['aws050wta'] / max(val)
    
    df['aws0100wta'] = df['aws0100wta'].fillna(0)
    val = list(set(df['aws0100wta']))
    df['aws0100wta'] = df['aws0100wta'] / max(val)
    
    df['aws0150wta'] = df['aws0150wta'].fillna(0)
    val = list(set(df['aws0150wta']))
    df['aws0150wta'] = df['aws0150wta'] / max(val)
    
    df['iccdcd'] = df['iccdcd'].fillna(0)
    val = list(set(df['iccdcd']))
    df['iccdcd'] = df['iccdcd'] / max(val)
    
    df['iccdcdpct'] = df['iccdcdpct'].fillna(0)
    val = list(set(df['iccdcdpct']))
    df['iccdcdpct'] = df['iccdcdpct'] / max(val)
    
    df['niccdcd'] = df['niccdcd'].fillna(0)
    val = list(set(df['niccdcd']))
    df['niccdcd'] = df['niccdcd'] / max(val)
    
    df['niccdcdpct'] = df['niccdcdpct'].fillna(0)
    val = list(set(df['niccdcdpct']))
    df['niccdcdpct'] = df['niccdcdpct'] / max(val)
    
    df['urbrecptwta'] = df['urbrecptwta'].fillna(0)
    val = list(set(df['urbrecptwta']))
    df['urbrecptwta'] = df['urbrecptwta'] / max(val)
    
    df['hydclprs'] = df['hydclprs'].fillna(0)
    val = list(set(df['hydclprs']))
    df['hydclprs'] = df['hydclprs'] / max(val)
    
    df['awmmfpwwta'] = df['awmmfpwwta'].fillna(0)
    val = list(set(df['awmmfpwwta']))
    df['awmmfpwwta'] = df['awmmfpwwta'] / max(val)
    
    # Construct adjacency matrix
    df_trg.sort()
    
    df = df.iloc[df_trg]
    
    topk = 3
    x, y = df['x'], df['y']
    
    tmp = []
    for i in range(len(x)):
        tmp.append([x.iloc[i], y.iloc[i]])
    
    tmp = torch.tensor(tmp)
    cdist = torch.cdist(tmp, tmp).to(device)
    trg_adj = torch.zeros(len(cdist), len(cdist)).to(device)
    for idx in range(len(cdist)):
        row = torch.topk(-cdist[idx], topk+1)
        trg_adj[idx] = torch.where(-cdist[idx] >= -cdist[idx][row[1][topk]], 1, 0)
    
    df = df.fillna(0)
    df = df.drop(columns=['wetland'])
    trg = df.drop(columns=['x', 'y'])
    
    # Return
    return src, trg, src_adj, trg_adj, src_label