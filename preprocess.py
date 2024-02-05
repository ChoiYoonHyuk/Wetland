def data_process(df):
    # Remove some columns with natural languages
    df = df.drop(columns=['OBJECTID', 'muname', 'mustatus', 'mukey', 'flodfreqmax', 'drclasswettest', 'engdwbdcd', 'engsldcd', 'flodfreqdcd', 'musym'])
    
    # Vectorize some columns with discrete values
    '''musym = list(set(df['musym']))
    for i in range(len(musym)):
        df['musym'] = df['musym'].replace(musym[i], i)'''
        
    drcls = list(set(df['drclassdcd']))
    for i in range(len(drcls)):
        df['drclassdcd'] = df['drclassdcd'].replace(drcls[i], i)
    
    hydgr = list(set(df['hydgrpdcd']))
    for i in range(len(hydgr)):
        df['hydgrpdcd'] = df['hydgrpdcd'].replace(hydgr[i], i)
        
    engdw = list(set(df['engdwobdcd']))
    for i in range(len(engdw)):
        df['engdwobdcd'] = df['engdwobdcd'].replace(engdw[i], i)
        
    engdwbll = list(set(df['engdwbll']))
    for i in range(len(engdwbll)):
        df['engdwbll'] = df['engdwbll'].replace(engdwbll[i], i)
    
    engdwbml = list(set(df['engdwbml']))
    for i in range(len(engdwbml)):
        df['engdwbml'] = df['engdwbml'].replace(engdwbml[i], i)
    
    engstafdcd = list(set(df['engstafdcd']))
    for i in range(len(engstafdcd)):
        df['engstafdcd'] = df['engstafdcd'].replace(engstafdcd[i], i)
    
    engstafll = list(set(df['engstafll']))
    for i in range(len(engstafll)):
        df['engstafll'] = df['engstafll'].replace(engstafll[i], i)
    
    engstafml = list(set(df['engstafml']))
    for i in range(len(engstafml)):
        df['engstafml'] = df['engstafml'].replace(engstafml[i], i)
    
    engsldcp = list(set(df['engsldcp']))
    for i in range(len(engsldcp)):
        df['engsldcp'] = df['engsldcp'].replace(engsldcp[i], i)
    
    englrsdcd = list(set(df['englrsdcd']))
    for i in range(len(englrsdcd)):
        df['englrsdcd'] = df['englrsdcd'].replace(englrsdcd[i], i)
    
    engcmssdcd = list(set(df['engcmssdcd']))
    for i in range(len(engcmssdcd)):
        df['engcmssdcd'] = df['engcmssdcd'].replace(engcmssdcd[i], i)
    
    engcmssmp = list(set(df['engcmssmp']))
    for i in range(len(engcmssmp)):
        df['engcmssmp'] = df['engcmssmp'].replace(engcmssmp[i], i)
    
    urbrecptdcd = list(set(df['urbrecptdcd']))
    for i in range(len(urbrecptdcd)):
        df['urbrecptdcd'] = df['urbrecptdcd'].replace(urbrecptdcd[i], i)
    
    forpehrtdcp = list(set(df['forpehrtdcp']))
    for i in range(len(forpehrtdcp)):
        df['forpehrtdcp'] = df['forpehrtdcp'].replace(forpehrtdcp[i], i)
    
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
    print(df.iloc[0])
    
    # Return
    return df