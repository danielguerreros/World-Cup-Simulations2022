# -*- coding: utf-8 -*-
"""
A collection of (poorly-commented) plotting and output routines

@author: @eightyfivepoint
"""

from re import M
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from PIL import Image
import urllib.request
from highlight_text import fig_text
import matplotlib.ticker as mtick

import urllib.request
import matplotlib
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
from plottable.plots import circled_image # image

def SetShortNames():
    # For plot axis labels
    ShortNames = {
        'Qatar':'QAT',
        'Brazil':'BRA',
        'Uruguay':'URU',
        'Argentina':'ARG',
        'South Korea':'SKR',
        'Iran':'IRN',
        'Japan':'JPY',
        'Saudi Arabia':'SAU',
        'Ecuador':'ECU',
        'Tunisia':'TUN',
        'Ghana':'GHA',
        'Senegal':'SEN',
        'Cameroon':'CAM',
        'Switzerland':'SWZ',
        'Serbia':'SER',
        'Portugal':'POR',
        'Spain':'SPA',
        'Croatia':'CRO',
        'Canada':'CAN',
        'Australia':'AUS',
        'Morocco':'MOR',
        'Denmark':'DEN',
        'France':'FRA',
        'Belgium':'BEL',
        'Wales':'WAL',
        'United States':'USA',
        'Poland':'POL',
        'Netherlands':'NET',
        'England':'ENG',
        'Germany':'GER',
        'Costa Rica':'COS',
        'Mexico':'MEX'
    }
    return ShortNames

def SetCodes():
    Codes = {
        'Qatar':5902,
        'Brazil':8256,
        'Uruguay':5796,
        'Argentina':6706,
        'South Korea':7804,
        'Iran':6711,
        'Japan':6715,
        'Saudi Arabia':7795,
        'Ecuador':6707,
        'Tunisia':6719,
        'Ghana':6714,
        'Senegal':6395,
        'Cameroon':6629,
        'Switzerland':6717,
        'Serbia':8205,
        'Portugal':8361,
        'Spain':6720,
        'Croatia':10155,
        'Canada':5810,
        'Australia':6716,
        'Morocco':6262,
        'Denmark':8238,
        'France':6723,
        'Belgium':8263,
        'Wales':394253,
        'United States':6713,
        'USA':6713,
        'Poland':8568,
        'Netherlands':6708,
        'England':8491,
        'Germany':8570,
        'Costa Rica':6705,
        'Mexico':6710,
        'Sweden':8520,
        'China':5822,
        'Nigeria':6346,
        'Scotland':8498,
        'South Africa':6316,
        'Thailand':5788,
        'Jamaica':5806,
        'Chile':9762,
        'Italy':8204,
        'New Zealand':5820,
        'Norway':8492,
        'Colombia':8258
    }
    return Codes

def SimWinners(sims,teamnames,save=True):
    ShortNames = SetShortNames()
    nTeamsPlot = 16 # number of teams to plot
    Nsims = len(sims)
    Codes = SetCodes()

    Winners = [x.KnockOut.Final[0].winner.name for x in sims]
    WinnerFreq = [(name,Winners.count(name)) for name in teamnames]
    WinnerFreq = sorted( WinnerFreq, key = lambda x : x[1], reverse=True)
    WinnerFreq = [(n,c) for (n,c) in WinnerFreq if c > 0]
    WinnerFreq = WinnerFreq[0:nTeamsPlot]
    WinnerNames = [x[0] for x in WinnerFreq]
    WinnerProp = np.array([x[1] for x in WinnerFreq],'float')/float(Nsims)

    #Create plot
    fig = plt.figure(figsize=(6, 2.5), dpi = 200)
    ax = plt.subplot(111)

    # Add spines
    ax.spines["top"].set(visible = False)
    ax.spines["right"].set(visible = False)

    # Add grid and axis labels
    ax.grid(True, color = "lightgrey", ls = ":")

    # We specify the width of the bar
    width = 0.5
    ax.bar(
        WinnerNames, 
        WinnerProp, 
        ec = "black", 
        lw = .75, 
        color = "#8a1538", 
        zorder = 3, 
        width = width,
        label = "Fouls conceded"
    )

    for index, y in enumerate(WinnerProp):
        ax.annotate(
            xy = (index, y),
            text = f"{y*100:.1f}%",
            xytext = (0, 7),
            textcoords = "offset points",
            size = 6,
            color = "#8a1538",
            ha = "center",
            va = "center",
            weight = "bold"
        )
    xticks = ax.xaxis.set_ticks(
        ticks = WinnerNames,
        labels = []
    )

    ax.tick_params(labelsize = 8)
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform

    # Native data to normalized data coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    fotmob_url = "https://images.fotmob.com/image_resources/logo/teamlogo/"
    for index, team_id in enumerate([Codes[x] for x in WinnerNames]):
        ax_coords = DC_to_NFC([index - width/2 - 0.16, -0.03])
        logo_ax = fig.add_axes([ax_coords[0], ax_coords[1], 0.09, 0.09], anchor = "W")
        club_icon = Image.open(urllib.request.urlopen(f"{fotmob_url}{team_id:.0f}.png"))
        logo_ax.imshow(club_icon)
        logo_ax.axis("off")


    fig_text(
        x = 0.12, y = 1.11,
        s = "Which National Teams Have More Chances to   \n Win the World Cup?",
        family = "DM Sans",
        weight = "bold",
        size = 10
    )

    fig_text(
        x = 0.12, y = 0.95,
        s = "viz by @sonofacorner | Model by @damaguesan",
        family = "Karla",
        color = "grey",
        size = 5
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    if save:
        figname = 'Plots/SimWinners.png'
        plt.savefig(figname,dpi=400,bbox_inches='tight',pad_inches=0.1) 
    #else:
    #    plt.show()
    
    
def SimFinalists(sims,teamnames,ShortNames):
   # Find most frequent finalists
   Finalists = [(x.KnockOut.Final[0].team1.name,x.KnockOut.Final[0].team2.name) for x in sims]
   F = [(x[0],x[1],Finalists.count(x)) for x in Finalists]
   F = sorted( F, key = lambda x : x[2], reverse=True)
   # get uniques
   FinalistFreq = []
   for f in F:
       if f not in FinalistFreq:
           FinalistFreq.append(f)
   print(FinalistFreq)

def TraceTeam(sims,teamname, verbose=False):
    # trace probability of a team progressing through the tournament
    Progress = []
    Nsims = float(len(sims))
    stages = ['GRP','R16','QF','SF','Final','Winner']
    for s in sims:
        p = 0
        if teamname in s.KnockOut.R16teamnames: p += 1
        if teamname in s.KnockOut.QFteamnames: p += 1
        if teamname in s.KnockOut.SFteamnames: p += 1
        if teamname in s.KnockOut.Finalteamnames: p += 1
        if teamname == s.KnockOut.Final[0].winner.name: p += 1
        Progress.append(stages[p])    
    ProgressFreq = [Progress.count(s)/Nsims for s in stages]
    assert np.isclose( np.sum(ProgressFreq),1.,atol=0.001,rtol=0.0)
    ProgressFreq = 1-np.cumsum(ProgressFreq)
    Progress = (teamname,ProgressFreq[0],ProgressFreq[1],ProgressFreq[2],ProgressFreq[3],ProgressFreq[4]) 
    if verbose:
        print("%s: %1.2f,%1.2f,%1.2f,%1.2f,%1.2f" % Progress)
    return Progress


def ExpectedGroupFinishes(sims,group_names, group_name):
    # Probability of each team in a group finishing in each position
    Nsims = len(sims)
    ind = group_names.index(group_name)
    Teams = sims[0].groups[ind].group_teams
    Table = {}
    for team in Teams:
        Table[team.name] = np.zeros(4)
    for i in range(0,Nsims):
        sims[i].groups[ind].build_table
        for t,p in zip(sims[i].groups[ind].table,range(0,4)):
            Table[t.name][p] += 1
    n = float(Nsims)
    Table = [(t,Table[t][0]/n,Table[t][1]/n,Table[t][2]/n,Table[t][3]/n) for t in Table.keys()]
    Table = sorted(Table,key = lambda x: x[1]+x[2],reverse=True)
    return Table
    
def ExpectedGroupFinishesPlot(sims,group_names,save=True):
    # Make group table probability plot
    Tables = [ExpectedGroupFinishes(sims,group_names,group) for group in group_names]
    fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(10, 9))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.45, hspace=0.4)
    nGroupTeams = 4
    for Table,ax,group in zip(Tables,axes.flatten(),group_names):
        grid = np.zeros((nGroupTeams,nGroupTeams),dtype=float)
        gridmax = 0.8
        gridmin = 0
        for i in range(nGroupTeams):
            for j in range(nGroupTeams):
                grid[i,j] = np.round( Table[i][j+1] ,3)
                if grid[i,j] <0.01:
                    grid[i,j] = gridmin
        Y = np.arange(nGroupTeams+0.5, 0, -1)
        X = np.arange(0.5, nGroupTeams+1, 1)
        X, Y = np.meshgrid(X, Y)
        cmap = plt.get_cmap('Blues')#cool, Reds, Purples
        levels = MaxNLocator(nbins=gridmax/0.01).tick_values(gridmin,gridmax)# grid.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(X, Y, grid,cmap=cmap,norm=norm)
        ax.set_xlim(0.5,nGroupTeams+0.5)
        ax.set_ylim(0.5,nGroupTeams+0.5)
        teams = [t[0] for t in Table]
        for i in range(nGroupTeams): 
            if teams[i]=='Korea Republic' or teams[i]=='South Korea':
                teams[i] = 'S. Korea'
        ax.set_yticks(np.arange(1,nGroupTeams+1,1))
        ax.set_xticks(np.arange(1,nGroupTeams+1,1))
        ax.set_yticklabels(teams[::-1],color='r',fontsize=11)
        ax.set_xticklabels( ['1st', '2nd', '3rd', '4th'], color='k', fontsize=11 )
        ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_title("Group " + group,color='r')
        pthresh=0.01
        Qual = np.array( np.round(100*np.sum( grid[:,:2], axis=1 ),0), dtype=int)
        for i in range(nGroupTeams):
            for j in range(nGroupTeams):
                if grid[i,j]>=pthresh:
                    s = "%1.0d" % (round(100*grid[i,j],0))
                    ax.text(j+0.9,nGroupTeams-i-0.15,s,fontsize=10,color='k')
        fig.set_facecolor('0.95')
        # twin axis
        ax2 = ax.twinx()
        ax2.set_ylim((0.5,0.5 + nGroupTeams ))
        ax2.set_yticks(np.arange(1,nGroupTeams+1,1))
        ax2.set_yticklabels( Qual[::-1] ,color='k')
        ax2.text(3.55+0.9,nGroupTeams+0.65,'Qual',fontsize=11,color='k')
        ax2.tick_params(axis=u'both', which=u'both',length=0)
    if save:
        figname = 'Plots/ExpectedGroupFinishes.png'
        plt.savefig(figname,dpi=400,bbox_inches='tight',pad_inches=0.1) 
        

def ExpectedGroupResults(sims,group_names, group_name):
    # Find most frequent results in group stage
    Nsims = len(sims)
    ind = group_names.index(group_name)
    resultslist = np.zeros( (Nsims,6),dtype = 'int')
    for i in range(0,Nsims):
        resultslist[i,:] = [100*m.team1_goals+m.team2_goals for m in sims[i].groups[ind].matches]
    most_freq = [ (int(x/100),int(x % 100)) for x in mode(resultslist)[0][0] ]
    # NOW PRINT RESULTS
    print(" GROUP %s RESULTS " % (group_name))
    for i in range(len(most_freq)):
        team1 = sims[0].groups[ind].matches[i].team1.name
        team2 = sims[0].groups[ind].matches[i].team2.name
        print("%s %s v %s %s" % (team1,most_freq[i][0],most_freq[i][1],team2))

def ExpectedKnockOutResults(sims,stage,Nmatches):
    # Find most frequent results in each knock-out match
    Nsims = float(len(sims))
    matches = 's.KnockOut.' + stage
    print(matches)     
    for i in range(Nmatches):
        resultslist = []
        for s in sims:
            m = eval(matches)
            resultslist.append((m[i].team1.name,m[i].team1_goals,m[i].team2.name,m[i].team2_goals))
        R = [(r[0],r[1],r[2],r[3],resultslist.count(r)/Nsims) for r in resultslist]
        R = sorted(R,key = lambda x: x[4], reverse=True)
        # get uniques
        ResultsFreq = []
        for r in R:
            if r not in ResultsFreq:
                ResultsFreq.append(r)
        # NOW PRINT RESULTS
        print(" KNOCKOUT RESULTS ") 
        for r in ResultsFreq[0:3]:
            print("%s,%s,%s,%s,%s" % r)


def makeProgressPlot( sims, teamnames, save=True ):
    # Probability of each team making it to each successive stage of the tournamnet
    ProgressArray = []
    for t in teamnames:
        ProgressArray.append( TraceTeam(sims,t) )
    nRounds = 5
    nteams = len( ProgressArray )
    ProgressArray = sorted( ProgressArray, key = lambda x: np.sum(x[1:]), reverse=True )
    grid = np.zeros((nteams,nRounds),dtype=float)
    gridmax = 0.9
    gridmin = 0
    for i in range(nteams):
        for j in range(nRounds):
            grid[i,j] = np.round( ProgressArray[i][j+1] ,3)
            if grid[i,j] <0.01:
                grid[i,j] = gridmin

    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10, 6))   
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.45)

    teams = [t[0] for t in ProgressArray]
    for i in range(nteams): 
        if teams[i]=='Korea Republic':
            teams[i] = 'South Korea'

    nteams = nteams/2

    for sp,ax in zip([0,1],axes):
        nteams=int(nteams)
    
        subgrid = grid[sp*nteams:(sp+1)*nteams,:]
        subteams = teams[sp*nteams:(sp+1)*nteams]
        Y = np.arange(nteams+0.5, 0, -1)
        X = np.arange(0.5, nRounds+1, 1)
        X, Y = np.meshgrid(X, Y)
        
        cmap = plt.get_cmap('Blues')#cool, Reds, Purples
        levels = MaxNLocator(nbins=gridmax/0.01).tick_values(gridmin,gridmax)# grid.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        ax.pcolormesh(X, Y, subgrid,cmap=cmap,norm=norm)
        ax.set_xlim(0.5,nRounds+0.5)
        ax.set_ylim(0.5,nteams+0.5)

        ax.set_yticks(np.arange(1,nteams+1,1))
        ax.set_xticks(np.arange(1,nRounds+1,1))
        ax.set_yticklabels(subteams[::-1],color='r',fontsize=11)
        ax.set_xticklabels( ['R16', 'QF', 'SF', 'F', 'W'], color='k', fontsize=12 )
        pthresh=0.01
        for i in range(nteams):
            for j in range(nRounds):
                if subgrid[i,j]>=pthresh:
                    s = "%1.0d" % (round(100*subgrid[i,j],0))
                    ax.text(j+0.9,nteams-i-0.1,s,fontsize=9,color='k')
                else:
                    ax.text(j+0.9,nteams-i-0.1,"<1",fontsize=9,color='k')
        fig.set_facecolor('0.95')
        ax2 = ax.twiny()
        ax2.set_xlim(0.5,nRounds+0.5)
        ax2.set_xticks(np.arange(1,nRounds+1,1))
        ax2.set_xticklabels( ['R16', 'QF', 'SF', 'F', 'W'], color='k' , fontsize=12 )
        ax.tick_params(axis='y',which='both',left='off',right='off')
        ax.tick_params(axis='x',which='both',top='off',bottom='off')
        ax2.tick_params(axis='x',which='both',top='off',bottom='off')
    fig.suptitle('WC2018: Probability of reaching round (%)',y=1.0,fontsize=14)
    if save:
        figname = 'Plots/ExpectedProgress.png'
        plt.savefig(figname,dpi=400,bbox_inches='tight',pad_inches=0.1)     

def TraceTeam_daniel(sims,teamname, verbose=False):
    # trace probability of a team progressing through the tournament
    Progress = []
    Nsims = float(len(sims))
    stages = ['GRP','R16','QF','SF','Final','Winner']
    for s in sims:
        p = 0
        if teamname in s.KnockOut.R16teamnames: p += 1
        if teamname in s.KnockOut.QFteamnames: p += 1
        if teamname in s.KnockOut.SFteamnames: p += 1
        if teamname in s.KnockOut.Finalteamnames: p += 1
        if teamname == s.KnockOut.Final[0].winner.name: p += 1
        Progress.append(stages[p])    
    ProgressFreq = [Progress.count(s)/Nsims for s in stages]
    assert np.isclose( np.sum(ProgressFreq),1.,atol=0.001,rtol=0.0)
    ProgressFreq = 1-np.cumsum(ProgressFreq)
    Progress = (teamname,ProgressFreq[0],ProgressFreq[1],ProgressFreq[2],ProgressFreq[3],ProgressFreq[4]) 
    data = {'Team':teamname,'Rd of 16':ProgressFreq[0],'Quarter finals':ProgressFreq[1],'SemiFinals':ProgressFreq[2],'Final':ProgressFreq[3],'WC':ProgressFreq[4]}
    dataframe = pd.DataFrame(data,index=[1])
    if verbose:
        print("%s: %1.2f,%1.2f,%1.2f,%1.2f,%1.2f" % Progress)
    return dataframe

def team_url(team_id):
    url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{team_id:.0f}.png"
#   np.array(PIL.Image.open(urllib.request.urlopen(url)))
    urllib.request.urlretrieve(url, f"Images/{team_id:.0f}.jpg")
    return(f"Images/{team_id:.0f}.jpg")

def expected_table(sims,teamnames,group_names,teamdata,save=True):
    Codes = SetCodes()
    
    Tables = [pd.DataFrame(ExpectedGroupFinishes(sims,group_names,group),columns=["Team","1st Prob","2nd Prob","3rd Prob","4th Prob"]) for group in group_names]
    data = pd.DataFrame()
    for t in teamnames:
        data=pd.concat([data,TraceTeam_daniel(sims,t)],ignore_index=True)
    
    df=teamdata.merge(pd.concat(Tables,ignore_index=True),on='Team').merge(data,on='Team').round(2).sort_values('Final',ascending=False).reset_index(drop=True)
    country_to_flagpath = {x: team_url(Codes[x]) for x in set(df["Team"])}
    df.insert(0, "Flag", df["Team"].apply(lambda x: country_to_flagpath.get(x)))
    df.set_index("Team",inplace=True)

    df=df[['Flag','Group','Elo', 'Moving goals for', 'Moving goals against',
       '1st Prob', '2nd Prob', '3rd Prob', '4th Prob', 'Rd of 16',
       'Quarter finals', 'SemiFinals', 'Final', 'WC']]
    df["Elo"]=df["Elo"].astype(int)
        
    cmap = LinearSegmentedColormap.from_list(
        name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )

    col_defs = (
    [
        ColumnDefinition(name="Flag",title="",textprops={"ha": "center"},width=0.5,plot_fn=circled_image,),
       ColumnDefinition(name="Team",textprops={"ha": "left", "weight": "bold"},width=1.5,), 
       ColumnDefinition(name="Group",textprops={"ha": "center"},width=0.75,),
       ColumnDefinition(name="Elo",group="Team Rating",textprops={"ha": "center"},width=0.75,),
       ColumnDefinition(name="Moving goals for",title="Goals \n for",width=0.75,textprops={"ha": "center","bbox": {"boxstyle": "circle", "pad": 0.35},},cmap=normed_cmap(df["Moving goals for"], cmap=matplotlib.cm.PiYG, num_stds=2.5),group="Team Rating",),
       ColumnDefinition(name="Moving goals against",title="Goals \n against",width=0.75,textprops={"ha": "center","bbox": {"boxstyle": "circle", "pad": 0.35},},cmap=normed_cmap(df["Moving goals against"], cmap=matplotlib.cm.PiYG_r, num_stds=2.5),group="Team Rating",),
       ColumnDefinition(name="1st Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",),
       ColumnDefinition(name="2nd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",),
#       ColumnDefinition(name="3rd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",),
#       ColumnDefinition(name="4th Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",),
       ColumnDefinition(name="Rd of 16",title="Make \n Rd of 16",formatter=decimal_to_percent,cmap=cmap,group="Knockout Stage Chances",border="left",),
       ColumnDefinition(name="Quarter finals",title="Make \n Quarters",formatter=decimal_to_percent,cmap=cmap,group="Knockout Stage Chances",border="left",),
       ColumnDefinition(name="Final",title="Make \n Finals",formatter=decimal_to_percent,cmap=cmap,group="Knockout Stage Chances",border="left",),
       ColumnDefinition(name="WC",formatter=decimal_to_percent,cmap=cmap,group="Knockout Stage Chances",border="left",),
       ColumnDefinition(name="SemiFinals",title="Make \n Semis",formatter=decimal_to_percent,cmap=cmap,group="Knockout Stage Chances",border="left",)
    ])

    fig, ax = plt.subplots(figsize=(15, 27))
    table = Table(
        df.drop(columns=["3rd Prob","4th Prob"]),
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        ax=ax,
        textprops={"fontsize": 14},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    ).autoset_fontcolors(colnames=["Moving goals for", "Moving goals against"])
    if save:
        fig.savefig("Plots/wwc_table.png", facecolor=ax.get_facecolor(), dpi=200)


def expectedGroups(sims,group_names,teamdata,save=True):
    cmap = LinearSegmentedColormap.from_list(
        name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )

    Codes = SetCodes()
    Tables = [pd.DataFrame(ExpectedGroupFinishes(sims,group_names,group),columns=["Team","1st Prob","2nd Prob","3rd Prob","4th Prob"]) for group in group_names]
    col_defs = (
        [
            ColumnDefinition(name="Flag",title="",textprops={"ha": "center"},width=0.5,plot_fn=circled_image,),
            ColumnDefinition(name="Team",textprops={"ha": "left", "weight": "bold"},width=1.5,), 
            ColumnDefinition(name="Group",textprops={"ha": "center"},width=0.75,),
            ColumnDefinition(name="Elo",textprops={"ha": "center"},width=0.75,),
            ColumnDefinition(name="1st Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="2nd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="3rd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="4th Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="Qual",formatter=decimal_to_percent,border="left",cmap=cmap)
        ]
    )

    fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(22, 17))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.05, hspace=0.05)

    for table,ax in zip(Tables,axes.flatten()):
            df=table.copy()
            df['Qual']=df.iloc[:,[1,2]].sum(axis=1)
            df=teamdata[["Team","Elo"]].merge(df,on='Team',how='right').round(2).sort_values('Qual',ascending=False).reset_index(drop=True)
            country_to_flagpath = {x: team_url(Codes[x]) for x in set(df["Team"])}
            df.insert(0, "Flag", df["Team"].apply(lambda x: country_to_flagpath.get(x)))
            df["Elo"]=df["Elo"].astype(int)
            df=df.set_index("Team")
            table = Table(
                    df,
                    column_definitions=col_defs,
                    row_dividers=True,
                    footer_divider=True,
                    ax=ax,
                    textprops={"fontsize": 14},
                    row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                    col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                    column_border_kw={"linewidth": 1, "linestyle": "-"},)
    if save:
        fig.savefig("Plots/groups_table.png", facecolor=ax.get_facecolor(), dpi=200)

def expectedGroup(sims,group,group_names,teamdata,save=True):
    cmap = LinearSegmentedColormap.from_list(
        name="bugw", colors=["#ffffff", "#f2fbd2", "#c9ecb4", "#93d3ab", "#35b0ab"], N=256
    )

    Codes = SetCodes()
    table = pd.DataFrame(ExpectedGroupFinishes(sims,group_names,group),columns=["Team","1st Prob","2nd Prob","3rd Prob","4th Prob"])
    col_defs = (
        [
            ColumnDefinition(name="Flag",title="",textprops={"ha": "center"},width=0.5,plot_fn=circled_image,),
            ColumnDefinition(name="Team",textprops={"ha": "left", "weight": "bold"},width=1.5,), 
            ColumnDefinition(name="Group",textprops={"ha": "center"},width=0.75,),
            ColumnDefinition(name="Elo",textprops={"ha": "center"},width=0.75,),
            ColumnDefinition(name="1st Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="2nd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="3rd Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="4th Prob",formatter=decimal_to_percent,group="Group Stage Chances",border="left",cmap=cmap),
            ColumnDefinition(name="Qual",formatter=decimal_to_percent,border="left",cmap=cmap)
        ]
    )

    fig,ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=0.05, hspace=0.05)

    
    df=table.copy()
    df['Qual']=df.iloc[:,[1,2]].sum(axis=1)
    df=teamdata[["Team","Elo"]].merge(df,on='Team',how='right').round(2).sort_values('Qual',ascending=False).reset_index(drop=True)
    country_to_flagpath = {x: team_url(Codes[x]) for x in set(df["Team"])}
    df.insert(0, "Flag", df["Team"].apply(lambda x: country_to_flagpath.get(x)))
    df["Elo"]=df["Elo"].astype(int)
    df=df.set_index("Team")
    table = Table(
                df,
                column_definitions=col_defs,
                row_dividers=True,
                footer_divider=True,
                ax=ax,
                textprops={"fontsize": 14},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},)
    if save:
        fig.savefig(f"Plots/group_table_{group}.png", facecolor=ax.get_facecolor(), dpi=200)
def simstats(sims): 
    # print some useful tournament stats
    print("Interesting simulation stats:")
    Favourites = ['Germany','Brazil','Spain','France','Argentina','Belgium','England']
    prev_winners = ['Germany', 'Brazil', 'Spain', 'France', 'Uruguay', 'Argentina', 'England']
    African = ['Egypt','Morocco','Nigeria','Senegal','Tunisia']
    SouthCentral = ['Brazil','Peru','Uruguay','Argentina','Mexico','Costa Rica','Panama','Colombia']
    Europe = ['Belgium','Serbia','England','France','Spain','Germany','Switzerland','Portugal','Sweden','Denmark','Poland','Croatia','Iceland']
    Australasia = ['Iran','Saudi Arabia','Russia','Australia','Japan','South Korea']
    All = African + SouthCentral + Europe + Australasia
    p = 0
    nsims = float(len(sims))
    for s in sims:
        if s.KnockOut.Final[0].winner.name not in prev_winners:
            p+=1
    print("New winners = %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if s.KnockOut.Final[0].winner.name  in All:
            p+=1
    #assert p==nsims
    p = 0
    for s in sims:
        if s.KnockOut.Final[0].winner.name  in Europe:
            p+=1
    print("Europe winners = %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if s.KnockOut.Final[0].winner.name  in African:
            p+=1
    print("African winners = %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if s.KnockOut.Final[0].winner.name  in Australasia:
            p+=1
    print("Australasia winners = %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if s.KnockOut.Final[0].winner.name  in SouthCentral:
            p+=1
    print("South Central American winners = %1.3f" % (p/nsims))
    p = 0   
    for s in sims:
        if s.KnockOut.groups[6].winner.name not in ['England', 'Belgium'] or s.KnockOut.groups[6].runner.name not in ['England', 'Belgium']:
            p+=1
    print("England or Beglium not qualify: %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if s.KnockOut.groups[7].winner.name in ['Japan', 'Senegal'] or s.KnockOut.groups[7].runner.name in ['Japan', 'Senegal']:
            p+=1
    print("Senegal or Japan qualify: %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if (s.KnockOut.R16matches[4].team1.name=='Brazil' and s.KnockOut.R16matches[4].team2.name=='Germany') or (s.KnockOut.R16matches[6].team1.name=='Germany' and s.KnockOut.R16matches[6].team2.name=='Brazil'):
            p+=1
    print("Brazil & Germany meet in R16: %1.3f" % (p/nsims))
    p = 0
    for s in sims:
        if (s.KnockOut.SFteamnames[0] not in Favourites) or (s.KnockOut.SFteamnames[1] not in Favourites) or (s.KnockOut.SFteamnames[2] not in Favourites) or (s.KnockOut.SFteamnames[3] not in Favourites):
            p+=1
    print("non-favourite makes semi-final: %1.3f" % (p/nsims))
