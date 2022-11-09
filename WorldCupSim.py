# -*- coding: utf-8 -*-
"""
Simulates the entire tournament once.

@author: @eightyfivepoint
"""

from WorldCupGroup import WorldCupGroup
from WorldCupKnockOut import WorldCupKnockOut

class WorldCupSim(object):
    def __init__(self,group_names,teams,verbose):
        self.group_names = group_names
        self.teams = teams
        self.groups = []
        self.verbose = verbose
        
    def runsim(self):
        # Run full World Cup Sim
        # Put teams in groups
        for g in self.group_names:
            group_teams = [t for t in self.teams if t.group==g]
            self.groups.append(WorldCupGroup(g,group_teams))
        # Simulation group matches
        for g in self.groups:
            if g.group_name=='A':
                #Match Qatar vs Ecuador.
                #Match Senegal vs Netherlands.
                #Match Qatar vs Senegal.
                #Match Ecuador vs Netherlands.
                #Match Netherlands vs Qatar.
                #Match Ecuador vs Senegal.
                pass
            elif g.group_name=='B':
                # Match Iran vs Wales.
                # Match United States vs England.
                # Match Iran vs United States.
                # Match Wales vs England.
                # Match England vs Iran.
                # Match Wales vs United States.
                pass
            elif g.group_name=='C':
                # Match Argentina vs Saudi Arabia.
                # Match Poland vs Mexico.
                # Match Argentina vs Poland.
                # Match Saudi Arabia vs Mexico.
                # Match Mexico vs Argentina.
                # Match Saudi Arabia vs Poland. 
                pass             
            elif g.group_name=='D':
                # Match Tunisia vs Australia.
                # Match Denmark vs France.
                # Match Tunisia vs Denmark.
                # Match Australia vs France.
                # Match France vs Tunisia.
                # Match Australia vs Denmark.
                pass
            elif g.group_name=='E':
                #Match Japan vs Spain.
                #Match Germany vs Costa Rica.
                #Match Japan vs Germany.
                #Match Spain vs Costa Rica.
                #Match Costa Rica vs Japan.
                #Match Spain vs Germany.
                pass
            elif g.group_name=='F':
                # Match Croatia vs Canada.
                # Match Morocco vs Belgium.
                # Match Croatia vs Morocco.
                # Match Canada vs Belgium.
                # Match Belgium vs Croatia.
                # Match Canada vs Morocco.
                pass
            elif g.group_name=='G':
                # Match Brazil vs Cameroon.
                # Match Switzerland vs Serbia.
                # Match Brazil vs Switzerland.
                # Match Cameroon vs Serbia.
                # Match Serbia vs Brazil.
                # Match Cameroon vs Switzerland.
                pass
            elif g.group_name=='H':
                # Match Uruguay vs South Korea.
                # Match Ghana vs Portugal.
                # Match Uruguay vs Ghana.
                # Match South Korea vs Portugal.
                # Match Portugal vs Uruguay.
                # Match South Korea vs Ghana.
                pass
            g.simulate_group_matches()
            if self.verbose:
                g.print_matches()
                g.print_table()
        # BUild knock-out stage of tournament
        self.KnockOut = WorldCupKnockOut(self.groups)
        # ROUND OF 16
        self.KnockOut.Round16()
        self.KnockOut.simulate_Round16_matches()
        if self.verbose:        
            self.KnockOut.print_matches(self.KnockOut.R16matches)
        # Quarter Finals
        self.KnockOut.QuarterFinal()
        self.KnockOut.simulate_QF_matches()
        if self.verbose:
            self.KnockOut.print_matches(self.KnockOut.QFmatches)
        # Semi Finals
        self.KnockOut.SemiFinal()
        self.KnockOut.simulate_SF_matches()
        if self.verbose:        
            self.KnockOut.print_matches(self.KnockOut.SFmatches)
        # Final
        self.KnockOut.Final()
        self.KnockOut.simulate_Final()
        if self.verbose:
            self.KnockOut.print_matches(self.KnockOut.Final)
            
