#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:42:06 2017

@author: monthubbardMacbookPro
"""
import numpy as np
from numpy import sin, cos
# 
def find_landing_transition_point(stateJ,thetaPs):
 
# jumper data
        xJ=stateJ[:,0];  
        vxJ=stateJ([:,1];
        yJ=stateJ[:,2];
        vyJ=stateJ[:,3];

        dydxJ=vyJ/vxJ; # dydx along design-speed jumper path
        thetaJ=arctan(dydxJ); # angle along design-speed jumper path

        # the landing transition point must happen downstream of point where the angle of
        # the jumper path is equal to the angle of the parent slope, meaning the jumper
        # is no longer moving upwards relative to the parent slope. 
        dydx_design=tan(-thetaPs); 
        xpara_interpolator = interp1d(dfydxJ, xJ,
                                  fill_value='extrapolate')
        xpara = xpara_interpolator(dydx_design)# x where jumper path slope is parallel to parent slope
        y_interpolator = interp1d(xJ, yJ,
                                  fill_value='extrapolate')
        ypara=y_interpolator(xpara)
        
        # we dont want transistion to start before that point  

        # cutting down parent slope data to limits of jumper path
        xP=xJ;
        yP=y_interpolator(xP)
        # figure ()
        # plot(xInrun,yInrun, xJ,yJ,xP,yP);axis equal;

        # goal is to find the last possible transition point (that by definition 
        # minimizes the transition snow budget) that satisfies the allowable 
        # transition G's
        # routine to find landing transition using Newton's method
        x = xpara; # first guess for transition point
        i = 0 ;# loop counter
        maxNumLoop = 1000 ;# maximum number of while loops
        gError = 1;# to get while loop started
        numpointsTranOut=100;
        delta=.01;# used for central difference approximation
        while gError >.001# tolerance
            #y=interp1(xJ,yJ,x); 
            y=y_interpolator(x) # y coordinate of design speed jumper path
            vx_interpolator = interp1d(xJ, vxJ,
                                  fill_value='extrapolate')
            vx=vx_interpolator(x) 
            #vx=interp1(xJ,vxJ,x); # velocity in x direction of jumper
            vy_interpolator = interp1d(xJ, vyJ,
                                  fill_value='extrapolate')
            vy=vy_interpolator(x) 
            #vy=interp1d(xJ,vyJ,x); # velocity in y direction of jumper
            v=sqrt(vx**2+vy**2);
            thetaJi=interp1(xJ,thetaJ,x); # angle of jumper path 
            yParenti=interp1(xPs,yPs,x); # y coordinate of parent slope
            dy = y-yParenti; # y coordinate difference between parent slope and jumper path
            vperp=sqrt(2*g*h) ;  # allowable perpendicular velocity
            #thetadiff = asin(vperp/v);# anle between safe landing and jumper path
            thetaLandi=thetaJi+asin(vperp/v); # allowable landing surface angle based on efh
            yprimeL = tan(thetaLandi);# dydx of safe landing surface
            yprimeT=tan(thetaLandi)+tan(thetaPs);# dydx of transition surface relative to parent slope( + tan(thetaPs))
            a=abs(dy/yprimeT); # required exponential characteristic distance, using three characteristic distances for transition
            # determining transition g's
            yprime=-dy/a - tan(thetaPs);# dydx of transition in global coordinate system
            # this is the same as the slope of the landing surface, tan(thetaLandi)
            #yprimeL=yprimeL;
            ydoubleprime=dy/a/a;# 2nd derivative of parent slope is zero
            k=abs(ydoubleprime/(1+yprime**2)**1.5); # curvature
            thetatot=atan(yprime);# this is the same as thetaLand
            transition_Gs=abs((k*v**2+g*cos(thetatot))/g); # transition g's

            gError = abs(transition_Gs-tolerableGs_tranOut);

        #     figure(50)
        #     plot(xP,yP,xJ,yJ,x,y,'+r',xpara,ypara,'ok'); 
        #     axis equal
            #pause
            x=x;
            dgdx=find_dgdx(x,stateJ,delta,xPs,yPs,h,g,thetaPs);
            dx=-gError/dgdx;
            x = x+dx;
            if x>xJ(end)
                x=xJ(end)-2*delta;
            end

            i = i+1;
            if i>maxNumLoop
                fprintf('ERROR: while loop ran more than #d times\n', maxNumLoop)
                break;
            end
        end

        x=x-dx;
        xTranOutEnd = x+3*a;# using three characteristic distances of exponential dcay
        xParent = linspace(x,xTranOutEnd,numpointsTranOut);
        yParent0 = interp1(xP,yP,x);
        yParent = yParent0-(xParent-x)*tan(thetaPs);
        #transition x,y information
        #xTranOut = x:dx:xTranOutEnd;# x coordinates of transition
        xTranOut = linspace(x,xTranOutEnd,numpointsTranOut);
        yTranOut = yParent+dy.*exp(-1.*(xTranOut-x)./a);# transition equation

    return x, y
