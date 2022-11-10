# -*- coding: utf-8 -*-
"""
Main script for running world cup simulations

@author: @eightyfivepoint
"""

import numpy as np
from WorldCupTeam import WorldCupTeam
from WorldCupSim import WorldCupSim
import WorldCupMetrics as met
import copy as copy
import pandas as pd
import random 

random.seed(40)
def get_WorldCup2022_data():
    teamdata = pd.read_excel('WorldCup2022_Teams.xlsx')
    return teamdata
                    
# ************ Simulations Parameters ***************
hostname = 'Qatar'
Nsims = 10# number of tournament simulations to run
verbose = False # set to true to print outcome of all matches and group tables in each sim iteration
savePlots = True # plots saved in same directory

# ************ Sim Set-up ***************
print("loading data")
teamdata = get_WorldCup2022_data()
group_names = sorted(np.unique(teamdata['Group']))
teamnames = list( teamdata['Team'].values )
sims = []

# ************ MAIN SIMULATIONS LOOP ***************
print("starting sims: 0 sims done")
for i in range(0,Nsims):
    # collect team data (needs to be redone in each loop of sim)
    teams = []
    for ix,row in teamdata.iterrows():
        teams.append( WorldCupTeam( row['Group'],row['Team'],row['Elo'],hostname,row["Moving goals for"],row["Moving goals against"]) )

    # initialise simulation
    s = WorldCupSim(group_names,teams,verbose=verbose)
    
    # run simulated world cup
    s.runsim() 
    sims.append(copy.deepcopy(s))

    if i>0 and i % 10 == 0: 
        print("               %s sims done" % (i))

# ************ Plots & Statistics ***************
print("generating plots and statistics")
met.SimWinners(sims,teamnames, save=savePlots)
met.ExpectedGroupFinishesPlot(sims,group_names, save=savePlots)
met.makeProgressPlot( sims, teamnames, save=savePlots )
met.expected_table(sims,teamnames,group_names,teamdata)
met.expectedGroups(sims,group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"A",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"B",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"C",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"D",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"E",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"F",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"G",group_names,teamdata,save=savePlots)
met.expectedGroup(sims,"H",group_names,teamdata,save=savePlots)
# Print some interesting tournament predictions
#met.simstats(sims)

print("done")

## ************  USEFUL INFO ************  ##

### to print tables for the jth group in the ith simulations 
#i=0 # 0->Nsims-1
#j=0 # group indices go from 0->7
#print sims[i].groups[j].print_table() 

### to print group matches 
#sims[i].groups[j].print_matches()

### to print results for a given KO round of the tournament 
#sims[i].KnockOut.print_matches(sims[i].KnockOut.R16matches)
#sims[i].KnockOut.print_matches(sims[i].KnockOut.QFmatches)
#sims[i].KnockOut.print_matches(sims[i].KnockOut.SFmatches)
#sims[i].KnockOut.print_matches(sims[i].KnockOut.Final)
        
### Print the most frequent results in each knock-out round
#met.ExpectedKnockOutResults(sims,'R16matches',8)
#met.ExpectedKnockOutResults(sims,'QFmatches',4)
#met.ExpectedKnockOutResults(sims,'SFmatches',2)
#met.ExpectedKnockOutResults(sims,'Final',1)