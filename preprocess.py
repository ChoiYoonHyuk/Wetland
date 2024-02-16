import pandas as pd
pd.options.mode.chained_assignment = None

def data_process(df):
    # Remove some columns with natural languages
    df = df.drop(columns=['OBJECTID', 'musym', 'mustatus', 'mukey', 'flodfreqmax', 'drclasswettest', 'engdwbdcd', 'engsldcd', 'flodfreqdcd'])
    
    # Vectorize some columns with discrete values
    feat = ['loam', 'clay', 'sand', 'fine', 'soil', 'complex', 'silt', 'coarse', 'outcrop', 'pit', 'flood', 'peat']
    #df[['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15']] = pd.DataFrame([[0] * 15], index=df.index)

    muname = list(set(df['muname']))
    for i in range(len(muname)):
        word = muname[i].replace(',', '').split(' ')
        for w in word:
            if w in feat:
                idx = feat.index(w)
                df.at[i, idx] = 1
    df = df.drop(columns=['muname'])
    
    for i in range(len(feat)):
        df[i] = df[i].fillna(0)
    
    
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
    
    df['brockdepmin'] = df['brockdepmin'].fillna(0)
    val = list(set(df['brockdepmin']))
    df['brockdepmin'] = df['brockdepmin'] / max(val)
    
    df['wtdepannmin'] = df['wtdepannmin'].fillna(0)
    val = list(set(df['wtdepannmin']))
    df['wtdepannmin'] = df['wtdepannmin'] / max(val)
    
    df['wtdepaprjunmin'] = df['wtdepaprjunmin'].fillna(0)
    val = list(set(df['wtdepaprjunmin']))
    df['wtdepaprjunmin'] = df['wtdepaprjunmin'] / max(val)
    
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
    #print(df.iloc[0])
    
    # Return
    return df