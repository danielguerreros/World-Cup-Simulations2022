# -*- coding: utf-8 -*-
"""
World Cup team class

@author: @eightyfivepoint
"""

class WorldCupTeam(object):
    def __init__(self,group,name,elo,hostname,moving_for,moving_against):
        self.name = name # Country
        self.group = group
        self.elorank = elo
        self.hometeam = name=='Qatar'
        self.group_matches = 0
        self.total_matches = 0
        self.points = 0
        self.goals_for = 0
        self.goals_against = 0
        self.host = self.name == hostname
        self.moving_for = moving_for
        self.moving_against = moving_against
              
        
    def __repr__(self):
        return "%s, %s, %s" % (self.name,self.group,self.elorank)