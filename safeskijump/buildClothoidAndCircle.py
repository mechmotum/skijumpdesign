#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:03:50 2017

@author: hubbard
"""
import numpy as np
from numpy import sin, cos


def buildClothoidAndCircle(lam,beta,vti,gamma,tolerableGs,g):
    """function [X,Y, dYdX, angle] = buildClothoidAndCircle(lam,beta,vti,gamma,tolerableGs,g)
    % This program will find the shape of a clothoid and circular transition
    % for the in-run of a ski jump with a circular section of radius, radius_min

    % lam = parent slope angle, radian
    % beta = takeoff angle, radians
    % lengthAp = length of the inrun approach section
    % vti = the velocity at the end of the approach section
    % gamma = percent of circular segment desired in transition

    % s1 = left side clothoid length at any point (downhill part)
    % s2 = right side clothoid at any point length (uphill part)
    % L1 = left side (longest) clothoid length (downhill part)
    % L2 = right side (longest) clothoid length (uphill part)

    Returns
    =======
    X : ndarray, shape(n,)
    Y : ndarray, shape(n,)
    dYdX : ndarray, shape(n,)
    angle : ndarray, shape(n,)

    """

    numpoints = 500
    rotation_clothoid = (lam-beta)/2
    # used to rotate symmetric clothoid so that left side is at lam and right sid is at beta

    # radius_min is the radius of the circular part of the transition.
    # Every other radius length (in the clothoid) will be longer than that, as this
    # will ensure the g-force felt by the skier is always less than a
    # desired value. This code ASSUMES that the velocity at the minimum
    # raidius is equal to the velocity at the end of the approach.
    radius_min = (vti**2)/(tolerableGs*g)

    #  x,y data for circle
    thetaCir = .5*gamma*(lam+beta)
    xCirBound = radius_min*sin(thetaCir)#s(end)+radius_min*sin(thetaCir)
    xCirSt = -radius_min*sin(thetaCir)#s(end)
    xCir = np.linspace(xCirSt, xCirBound,numpoints)
    yCir = radius_min-np.sqrt(radius_min**2-xCir**2)

    ## x,y data for clothoid
    A_squared = (radius_min**2)*(1-gamma)*((lam+beta)) # for one clothoid
    A = np.sqrt(A_squared)
    clothoid_length = A*np.sqrt((1-gamma)*(lam+beta))# length of one clothoid
    #checkClothoidLength = radius_min*(1-gamma)*(lam+beta)
    #total_transition_length = 2*clothoid_length# two clothoid linked together at rmin
    #[clothoid_length:-0.1:0]
    s = np.linspace(clothoid_length,0,numpoints)# generates arc length points for one clothoid
    X1 = s-(s**5)/(40*A**4)+(s**9)/(3456*A**8)# x coordinate for the clothoid shape
    Y1 = (s**3)/(6*A**2)-(s**7)/(336*A**6)+(s**11)/(42240*A**10)# y coordinate for the clothoid shape
    #plot(X1,Y1,'r','linewidth',3.0) axis equal hold on#X1 is black
    # axis equal
    # grid on
    # plot (0,radius_min, 'x') # this 'x' marks the center of the circle
    # legend('X1')

    X2 = X1-X1[0]
    #X2 = X1+X1(end)-X1(1)
    Y2 = Y1-Y1[0]
    #plot(X2,Y2,'b','linewidth',3.0) axis equal #X2 is blue
    # this shifts the original clothoid so that the origin is correct
    #the curved end of the clothoid is now at the origin(0,0)
    #pause

    theta = (lam+beta)/2
    X3 = (cos(theta)*X2)+(sin(theta)*Y2)
    Y3 = (-sin(theta)*X2)+(cos(theta)*Y2)
    # this rotates the clothoid to theta
    # theta   =  lam+beta/2 where beta and lam are positive

    #plot(X3,Y3,'m','linewidth',2.0) axis equal #X3 is red
    #pause
    X4 = X3
    Y4 = Y3#-radius_min
    #plot(X4,Y4,'c','linewidth',2.0) axis equal #X4 is thin red
    # this shifts the y-axis so that it can be rotated about the new origin
    #
    X5 = -X4+2*X4[0]
    Y5 = Y4#plot(X5,Y5,'g','linewidth',2.0) axis equal #X5 is thin green
    # this does the same for the right side
    # THE RIGHT AND LEFT SIDES ARE MIRROR IMAGES OF EACH OTHER

    X4 = X4-radius_min*sin(thetaCir)
    Y4 = Y4+radius_min*(1-cos(thetaCir))
    X4 = X4[::-1]
    Y4 = Y4[::-1]

    X5 = X5+radius_min*sin(thetaCir)
    Y5 = Y5+radius_min*(1-cos(thetaCir))

    #figure()
    #plot(X5,Y5,'m',X4,Y4,'b') axis equal hold on
    ## stitching together clothoid and circular data
    xLCir = xCir[xCir< = 0]
    yLCir = radius_min-np.sqrt(radius_min**2-xLCir**2)

    xRCir = xCir[xCir> = 0]
    yRCir = radius_min-np.sqrt(radius_min**2-xRCir**2)

    X4 = np.hstack((X4, xLCir[1:-1]))
    Y4 = np.hstack((Y4, yLCir[1:-1]))

    X5 = np.hstack((xRCir[0:-2], X5))
    Y5 = np.hstack((yRCir[0:-2], Y5))

    # figure()
    # plot(X5,Y5,'b',X4,Y4,'k') hold onaxis equal
    # plot(xRCir(end),yRCir(end),'or')
    # plot(xLCir(1),yLCir(1),'ok')
    # title('clothoid to show percent circle')
    # hold off

    X6 = (cos(rotation_clothoid)*X4)+(sin(rotation_clothoid)*Y4)
    Y6 = (-sin(rotation_clothoid)*X4)+(cos(rotation_clothoid)*Y4)
    #figure()
    #plot(X6,Y6,'c','linewidth',2.0) axis equal hold on
    # this rotates the left side at the correct angle (lam)
    # AND THE DATA POINTS SHIFT TO THE LEFT
    #pause
    X7 = (cos(rotation_clothoid)*X5)+(sin(rotation_clothoid)*Y5)
    Y7 = (-sin(rotation_clothoid)*X5)+(cos(rotation_clothoid)*Y5)
    #plot(X7,Y7,'y','linewidth',2.0) axis equal #X7 is yellow
    #this rotates the RIGHT side at the correct angle (lam2)
    # rotation   =  (lam-beta)/2 where lam and beta are both positive
    #pause

    X  =  np.hstack((X6,  X7))
    Y  =  np.hstack((Y6,  Y7))
    dYdX  =  np.diff(Y)/np.diff(X)
    angle  =  np.arctan(dYdX)
    # make plot to visualize where circle and clothoid are
    xCirBound = np.array((xRCir[-1],xLCir[0]))
    yCirBound = np.array((yRCir[-1],yLCir[0]))
    xCirBoundRot = (cos(rotation_clothoid)*xCirBound)+(sin(rotation_clothoid)*yCirBound)
    yCirBoundRot = (-sin(rotation_clothoid)*xCirBound)+(cos(rotation_clothoid)*yCirBound)
    #
    # figure ()
    # plot (X,Y,'g','LineWidth',4) axis equal hold on
    # plot (xCirBoundRot, yCirBoundRot,'xk',...
    #     'MarkerSize',15,'LineWidth',5)
    # ylabel('Vertical distance (m)')
    # xlabel ('Horizontal distance (m)')
    # title ('X,Y Plot of Takeoff Transition')
    # legend ('Takeoff Transition','Bounds of circular segement')
    # hold off
    #

    #end

    #return xTrCloAndCir,yTrCloAndCir, slopeClo, angleClo
    return  X,Y, dYdX, angle

X, Y, dYDX, angle = buildClothoidAndCircle(0.3,0.3,20,0.99,1.5,9.81)
