# -*- coding: utf-8 -*-
"""
World Cup match class. Simulates the individual matches.

@author: @eightyfivepoint
"""

import scipy as scipy
import scipy.stats
import numpy as np
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle
from scipy.stats import poisson
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

modelo_poisson = load_pickle("Prediction Model/international_model.pickle")
class WorldCupMatch(object):
    def __init__(self,team1,team2):
        self.played = False
        self.penalties = False
        self.team1 = team1
        self.team2 = team2
        self.eloK = 60 # for world cup (see http://www.eloratings.net/about)
        self.run_hot = True
        self.elo_only_model()
        
    def __repr__(self):
        if self.played:
            return "Match %s %s vs %s %s" % (self.team1.name, self.team1_goals, self.team2_goals, self.team2.name)
        else:
            return "Match %s vs %s." % (self.team1.name, self.team2.name)
        
    def elo_only_model(self):
        # log( mu ) = const + xe*(elo_diff/100) + eloHA*IsHost?
        self.const = 0.1557
        self.xe = 0.169 # coeff elo_diff / 100
        self.eloHA = 0.182/self.xe*100 # get regression coefficient in units of elo difference
        self.model = "Elo"
            
    def set_group_stats(self,team1_goals, team2_goals, stage):
        # update 
        self.played = True
        if stage=='GRP':
            self.team1.group_matches += 1
            self.team2.group_matches += 1
        self.team1.total_matches += 1
        self.team2.total_matches += 1
        self.team1_goals = team1_goals
        self.team2_goals = team2_goals
        self.team1.goals_for += team1_goals
        self.team1.goals_against += team2_goals
        self.team2.goals_for += team2_goals
        self.team2.goals_against += team1_goals       
        self.team1.goal_dif = self.team1.goals_for - self.team1.goals_against
        self.team2.goal_dif = self.team2.goals_for - self.team2.goals_against   
        if self.team1_goals > self.team2_goals:
            self.winner = self.team1
            if stage=='GRP':
                self.team1.points += 3                        
        elif self.team1_goals < self.team2_goals:
            self.winner = self.team2            
            if stage=='GRP':
                self.team2.points += 3
        else:
            if stage=='GRP':
                self.team1.points += 1
                self.team2.points += 1
     
    
    def generate_result(self,stage,penalties=False,verbose=False):
        
        # Elo differences, including home advantage effect
        elo_diff = self.team1.elorank - self.team2.elorank
        
        if self.model=="Elo":
            mu1 =  modelo_poisson.predict(pd.DataFrame(data={'mov_score_for': self.team1.moving_for,  'mov_score_against': self.team1.moving_against,'mov_score_against_rival':self.team2.moving_against,'home':int(self.team1.host),"Elo_diff":self.team1.elorank - self.team2.elorank},index=[1])).values[0]
            mu2 =  modelo_poisson.predict(pd.DataFrame(data={'mov_score_for': self.team2.moving_for,  'mov_score_against': self.team2.moving_against,'mov_score_against_rival':self.team1.moving_against,'home':int(self.team2.host),"Elo_diff":self.team2.elorank - self.team1.elorank},index=[1])).values[0]
        # Draw goals from Poisson distribution
        team1_goals = int(scipy.stats.poisson.rvs(mu1,size=1))
        team2_goals = int(scipy.stats.poisson.rvs(mu2,size=1))
        if verbose:
            print("%s %d vs %d %s" % (self.team1.name, team1_goals, team2_goals, self.team2.name))
            print( "Mus: %1.1f, %1.1f" % (mu1,mu2))
            print("Elos: %1.1f, %1.1f" % (self.team1.elorank,self.team2.elorank))
        # Update group stats
        self.set_group_stats(team1_goals,team2_goals,stage)
        # Simulate penalty shoot-out if necessary
        if penalties and team1_goals==team2_goals: 
            self.penalty_shootout()
        # Finally, update elo score for each team if running 'hot'
        if self.run_hot:
            self.update_elo_scores_ELORATING( team1_goals-team2_goals, elo_diff)
        if verbose:
            print("Elos: %1.1f, %1.1f" % (self.team1.elorank,self.team2.elorank))

    def generate_probabilities(self):
        # Elo differences, including home advantage effect
        elo_diff = self.team1.elorank - self.team2.elorank
        
        #if self.model=="Elo":
        mu1 =  modelo_poisson.predict(pd.DataFrame(data={'mov_score_for': self.team1.moving_for,  'mov_score_against': self.team1.moving_against,'mov_score_against_rival':self.team2.moving_against,'home':int(self.team1.host),"Elo_diff":elo_diff},index=[1])).values[0]
        mu2 =  modelo_poisson.predict(pd.DataFrame(data={'mov_score_for': self.team2.moving_for,  'mov_score_against': self.team2.moving_against,'mov_score_against_rival':self.team1.moving_against,'home':int(self.team2.host),"Elo_diff":-elo_diff},index=[1])).values[0]
        # Draw goals from Poisson distribution
        
        team_pred = [[poisson.pmf(i, team_avg) for i in range(0, 6)] for team_avg in [mu1, mu2]]
        return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

    
    def generate_probability_plot(self,save=True):
        partido=self.generate_probabilities()
        local = self.team1.name
        visitante = self.team2.name
        series=pd.Series([1-np.sum(np.triu(partido, 1))-np.sum(np.diag(partido)),np.sum(np.diag(partido)),np.sum(np.triu(partido, 1))],index=[f'{local}','Draw',f'{visitante}'])

        color_xpecta='#7ee1bd'
        myColor=LinearSegmentedColormap.from_list("mio",['#FFFFFF',color_xpecta],N=100)

        fig = plt.figure(figsize=(6, 7))
        gs = GridSpec(nrows=3, ncols=2, width_ratios=[3, 1], height_ratios=[1, 4, 1])   

        # First axes
        ax0 = fig.add_subplot(gs[1, 0])
        g3 = sns.heatmap(partido*100,cbar=False,annot=True,fmt = '.1f',square=1,cmap=myColor,ax=ax0)
        for t in ax0.texts: t.set_text(t.get_text() + " %")
        ax0.set_xlabel(f"{visitante}")
        ax0.set_ylabel(f"{local}")

        # Second axes
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar([n for n in range(6)],partido.sum(axis=0)*100,color=color_xpecta)
        #ax1.set_yticks(np.arange(len(partido.sum(axis=0)*100)))
        ax1.set_axisbelow(True)
        ax1.grid(axis='y')


        # Third axes
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.barh([n for n in range(6)],partido.sum(axis=1)*100,color=color_xpecta)
        ax2.set_axisbelow(True)
        ax2.grid(axis='x')
        ax2.invert_yaxis()


        ax3 = fig.add_subplot(gs[2, :])
        np.round(pd.DataFrame(series).T*100,2).plot.barh(stacked=True,ax=ax3)
        ax3.axis('off')
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 0.3),ncol=3,frameon=False)
        
        for p in ax3.patches:
            left, bottom, width, height = p.get_bbox().bounds
            if width > 0:
                ax3.annotate(f'{width:0.0f}%', xy=(left+width/2, bottom+height/2), ha='center', va='center',color='black')

        ax4 = fig.add_subplot(gs[0,1])
        #ax4.imshow("Images/5796.jpg")
        ax4.axis('off')


        plt.suptitle(f"Probabilidades {local} vs {visitante} ",y=0.92,fontsize=14)
        plt.figtext(0.62,0.1, f"Ganador más probable: {series.idxmax()} ({np.round(series.max()*100,1)}%)", ha="right", va="top", fontsize=12, color="w")
        plt.figtext(0.612,0.06, "Resultado más probable:"+f" {np.argmax(np.max(partido, axis=1))}-{np.argmax(np.max(partido, axis=0))} ({np.round(partido[np.argmax(np.max(partido, axis=1))][np.argmax(np.max(partido, axis=0))]*100,1)}%)", ha="right", va="top", fontsize=12, color="w")
        plt.figtext(0.5,0.01, "Modelo y Visualización: @damaguesan", ha="left", va="top", fontsize=9, color="w")

        if save:
            fig.savefig(f"Plots/{local}{visitante}Probability.png",  dpi=200) 
            
    def update_elo_scores_ELORATING(self, goal_diff, elo_diff):
        # see http://www.eloratings.net/about
        if np.abs( goal_diff ) == 2:
            K = self.eloK * 1.5
        elif np.abs( goal_diff ) > 2:
            K = self.eloK * (1.75 + ( np.abs( goal_diff ) - 3 ) / 8. )
        else:
            K = self.eloK
        W = 1. if goal_diff>0 else 0. if goal_diff<0 else 0.5
        We = 1. / ( 10**(-1.*elo_diff/400.) + 1. )        
        self.team1.elorank += K * (W-We)
        self.team2.elorank += K * (We-W)


    def penalty_shootout(self, Ninit = 5):
        # generate 5 penalties each and check against skill
        self.penalties = True
        team1_success = np.sum(np.random.uniform(size=Ninit))
        team2_success = np.sum(np.random.uniform(size=Ninit))
        if team1_success > team2_success:
            self.winner = self.team1
        elif team1_success < team2_success:
            self.winner = self.team2
        #else: # determine winner based on relative penalty strenghts
        #    high = self.team1.penaltyskill + self.team2.penaltyskill
        #    if np.random.uniform(size=1,high=high)<self.team1.penaltyskill:
        #        self.winner = self.team1
        #    else:
        #        self.winner = self.team2


    
    
