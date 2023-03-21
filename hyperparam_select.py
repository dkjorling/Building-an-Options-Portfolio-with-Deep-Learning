# find average rank of all pairs
N_Heads = [4, 8]
Dim_Vals = [1024, 2048, 4096]
rank_dict = {'N_Heads': [],
            'Dim_Vals': [],
            'Avg_Rank': [],
            'Avg_Loss': []}
for n in N_Heads:
    for d in Dim_Vals:
        rank_dict['N_Heads'].append(n)
        rank_dict['Dim_Vals'].append(d)
        ranks = []
        avl = []
        for y in dfp['Last_Train_Year']:
            df2 = df[df['Last_Train_Year'] == y].sort_values('Average_Validation_Loss').reset_index()
            vl = df2[(df2['Dim_Vals'] == d) & (df2['N_Heads'] == n)]['Average_Validation_Loss']
            rank = df2[(df2['Dim_Vals'] == d) & (df2['N_Heads'] == n)].index[0] + 1
            avl.append(vl)
            ranks.append(rank)
        rank_dict['Avg_Rank'].append(np.mean(ranks))
        rank_dict['Avg_Loss'].append(np.mean(avl))
rank_df = pd.DataFrame(rank_dict).sort_values('Avg_Rank', ascending=True)