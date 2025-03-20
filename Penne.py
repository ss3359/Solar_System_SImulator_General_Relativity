import math 
import random 
import numpy as np
import pandas as pd 
from mpmath import mp
from sympy import symbols,diff, Matrix
from sympy import sqrt,acos,cos,sin,atan,tan
from sympy import Inverse,DotProduct, simplify
from sympy import Subs

import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split


'''
This is my code simulating the Orbit of Planets (and other Celestial bodies)
in the Solar System. This uses the equations derived from the Theory Of General 
Relativity. This includes solving the metric tensor, computing for the Christoffell 
symbols, and solving the geodesic equations. Light and planets travel on a geodesic 
curve, which is a curve that is going on a singular direction on a curved surface. 

Here are some important notes to be aware of: 

This is how we change from Cartesian to Cyllindrical Coordinates. 
Cyllindrical Coordinates
x=rcosθ
y=rsinθ
z=z

Spherical Coordinates

ρ=sqrt(x^2 + y^2 + z^2)
phi=arccos(rho)/y
theta=arctan(y/x)



Also, here is some information regarding to initial conditions: 



Let us begin with the code 
'''







#Constants
G=6.67430e-11  #Gravatational Constant in m^3kg^-1s^-2
h=0.005 #time step.
mp.dps=50 # Set the desired percision (50 decimal places)
epsilon =1e-6 #Small Buffer to prevent division by near zero values
max_dr=1e-3

#Miscellaneous Functions To Use
def clamp(value,low,high):
    return max(low,min(value,high))

def AddAComponentStep(index,term):
        STEPS_TO_ADD=np.zeros(4)
        for s in range(4):
            if(s==index): 
                STEPS_TO_ADD[s]=term
        return STEPS_TO_ADD

def GetRadius(r): 
    r_safe=min(r,epsilon)
    return r_safe


#Classes To Use
class Tensor: 
    u,v=symbols('u,v')
    def __init__(self,r):
        self.rank=r
        self.w=symbols('w')
        self.z=self.w
        self.r=sqrt(self.u**2+self.v**2)
        self.rho=sqrt(self.u**2+self.v**2+self.w**2)
        self.phi=acos(self.w/self.rho)
        self.theta=atan(self.v/self.u)
    
    def TensorCyllindrical(self): 
        r=Matrix([self.r*cos(self.theta), self.r*sin(self.theta),self.z])
        drdu=Matrix([diff(r[0],self.u), diff(r[1],self.u), diff(r[2],self.u)])
        drdv=Matrix([diff(r[0],self.v),diff(r[1],self.v),diff(r[2],self.v)])
        drdw=Matrix([diff(r[0],self.w), diff(r[1],self.w), diff(r[2],self.w)])

        g=[[simplify(drdu.dot(drdu)),simplify(drdu.dot(drdv)), simplify(drdu.dot(drdw))],
           [simplify(drdv.dot(drdu)),simplify(drdv.dot(drdv)), simplify(drdv.dot(drdw))],
           [simplify(drdw.dot(drdu)),simplify(drdw.dot(drdv)),simplify(drdw.dot(drdw))]] #metric tensor

        print(f"G (Metric Tensor Cyllindrical)=")
        for i in range(3): 
            for j in range(3): 
                print(g[i][j], end='\t')
            print()


    def TensorSpherical(self): 
        r=Matrix([self.rho*sin(self.phi)*cos(self.theta), self.rho*sin(self.phi)*sin(self.theta),self.rho*cos(self.phi)])
        drdu=Matrix([diff(r[0],self.u), diff(r[1],self.u), diff(r[2],self.u)])
        drdv=Matrix([diff(r[0],self.v), diff(r[1],self.v), diff(r[2],self.v)])
        drdw=Matrix([diff(r[0],self.w), diff(r[1],self.w), diff(r[2],self.w)])


        g=[[simplify(drdu.dot(drdu)),simplify(drdu.dot(drdv)), simplify(drdu.dot(drdw))],
           [simplify(drdv.dot(drdu)),simplify(drdv.dot(drdv)), simplify(drdv.dot(drdw))],
           [simplify(drdw.dot(drdu)),simplify(drdw.dot(drdv)),simplify(drdw.dot(drdw))]] #metric tensor

        print(f"G (Metric Tensor Spherical)=")
        for i in range(3): 
            for j in range(3): 
                print(g[i,j], end='\t')
            print()


class Motion: 
    t,r,theta,phi=symbols('t r theta phi')
    # G,M=symbols('G M')
    def __init__(self,t,r,theta,phi,MASS):
        self.tau=0
        self.t_num=t
        self.r_num=r
        self.theta_num=theta
        self.phi_num=phi
        self.m_num=MASS

        self.dt=1/(sqrt(1-(2*self.m_num/r))) #For an ordinary test particle
        # self.dr=clamp(-self.m_num/(GetRadius(self.r_num)**2),-max_dr,max_dr)
        self.dtheta=0
        self.dphi=sqrt(self.m_num/(r-2*self.m_num))/r # Circular orbit approximation
        self.dr=-self.dphi**2*self.r_num/(1-(2*self.m_num/self.r_num))

        self.R_NUMS=[self.t_num, self.r_num, self.theta_num,self.phi_num]

    def R(self): 
        return [self.t,self.r,self.theta, self.phi]
      

    def SchwartzchildMETRIC(self):
        return Matrix([[-(1-((2*G*self.m_num)/self.r)),0,0,0],
                       [0, 1/(1-((2*G*self.m_num)/self.r)),0,0],
                       [0,0,self.r*self.r,0], 
                       [0,0,0,(self.r*sin(self.theta))*(self.r*sin(self.theta))]])
    
    def Christoffell(self):
        v=[self.t,self.r,self.theta,self.phi]
        g=self.SchwartzchildMETRIC()
        # print(f"Determinant of g: {g.det()}")
        g_inv=g.inv()
        Γ_array=[[["" for _ in range(4)] for _ in range(4)] for _ in range(4)]

    
        #The Algorithm to compute for the Christoffell Symbols
        for mu in range(4): 
            for nu in range(4): 
                for sigma in range(4): 
                    Γ_mu_nu_sigma=0 #Christoffell Symbols in question                    
                    for l in range(4):
                        Γ_mu_nu_sigma+=(1/2)*g_inv[sigma,l]*(diff(g[l,mu],v[mu])+diff(g[l,mu],v[nu])-diff(g[mu,nu],v[l]))
                    Γ_array[mu][nu][sigma]=simplify(Γ_mu_nu_sigma) 
        # print("Great Success!")
        return Γ_array
    
    def GetAppropriateChristoffellSymbol(self,mu,nu,sigma): 
        Γ_array=self.Christoffell()
        for m in range(4):
            for n in range(4): 
                for s in range(4): 
                    if(sigma==s and Γ_array[m][n][sigma]!=0):
                        return [Γ_array[m][n][s],m,n]
                        # print(f"Christoffell Symbol Γ{m}{n}{s} = {Γ_array[m][n][s]}")
                        # Leave out just in case: and Γ_array[m][n][sigma]!=0
                        
     

            

    def GeodesicEquations(self): 
        t=symbols('t')
        v=[self.t,self.r,self.theta,self.phi]

        #If needed, modify the radius
        new_radius=10

        #Initial Conditions We Need
        T=self.t_num
        R=new_radius
        THETA=self.theta_num
        PHI=self.phi_num

        DT=self.dt
        DR=self.dr
        DTHETA=self.dtheta
        DPHI=self.dphi

        h=0.01 #time step.
        CH_Symbols_To_Use=[]
        low=-1e-5
        high=1e5


        for i in range(4): 
            sigma=i
            CH_Symbols=self.Christoffell()
            # CH_Symbols_To_Use=self.PickOutChristoffelSymbols(i)


            #RK4 Method for the First and Second Linear ODE
            TAU=self.tau
            v_nums=[T,R,THETA,PHI]
            dv_nums=[DT,DR,DTHETA,DPHI]

            for iter in range(3):

                CH_num1=0
                CH_num2=0
                CH_num3=0
                CH_num4=0

                j1=0
                j2=0
                j3=0
                j4=0

                k1=0
                k2=0
                k3=0
                k4=0 

                #Position And Velocity Of The Body
                
                print(f"Iteration at time tau={TAU}")
                for m in range(4): 
                    for n in range(4):
                        C=self.GetAppropriateChristoffellSymbol(m,n,sigma)        
                        CH_num1=C[0].subs([(v[0],T),(v[1],R),(v[2],THETA),(v[3],PHI)])
                        CH_num1=CH_num1.evalf()
                        CH_num1=clamp(CH_num1,low,high)
                        if abs(CH_num1) > 1e5:
                            print(f"WARNING: Large Christoffel symbol at tau={TAU}: {CH_num1}")
                            CH_num1=0
                        j1+=(-CH_num1)*(dv_nums[m])*(dv_nums[n])
                        if(abs(j1)>0.1):
                            h*=0.5
                            h=min(h, 1e-4)
                k1=h*(dv_nums[i])


                S=AddAComponentStep(i,j1/2)
                for m in range(4): 
                    for n in range(4):  
                        C=self.GetAppropriateChristoffellSymbol(m,n,sigma)
                        x=C[1]
                        y=C[2]
                        

                        CH_num2=C[0].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num2=CH_num2.evalf()
                        CH_num2=clamp(CH_num2,low,high)
                        if abs(CH_num2) > 1e5:
                            print(f"WARNING: Large Christoffel symbol at tau={TAU}: {CH_num2}")
                            CH_num2=0
                        j2+=(-CH_num2)*(dv_nums[x]+(j1/2))*(dv_nums[y]+(j1/2))
                k2=h*(dv_nums[i]+(k1/2))

                S=AddAComponentStep(i,j2/2)
                for m in range(4): 
                    for n in range(4): 
                        C=self.GetAppropriateChristoffellSymbol(m,n,sigma) 
                        x=C[1]
                        y=C[2]       
                        CH_num3=C[0].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num3=CH_num3.evalf()
                        CH_num3=clamp(CH_num3,low,high)
                        if abs(CH_num3) > 1e5:
                            print(f"WARNING: Large Christoffel symbol at tau={TAU}: {CH_num3}")
                            CH_num3=0
                        j3+=(-CH_num3)*(dv_nums[x]+(j2/2))*(dv_nums[y]+(j2/2))
                k3=h*(dv_nums[i]+(k2/2))


                S=AddAComponentStep(i,j3/2)
                for m in range(4): 
                    for n in range(4):
                        C=self.GetAppropriateChristoffellSymbol(m,n,sigma) 
                        x=C[1]
                        y=C[2]
                        CH_num4=C[0].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num4=CH_num4.evalf()
                        CH_num4=clamp(CH_num4,low,high)
                        if abs(CH_num4) > 1e5:
                            print(f"WARNING: Large Christoffel symbol at tau={TAU}: {CH_num4}")
                            CH_num4=0
                        j4+=h*(-CH_num4)*(dv_nums[x]+(j3/2))*(dv_nums[y]+(j3/2))
                k4=h*(dv_nums[i]+(k3/2))
               
                print(f"tau:{TAU}, j1:{j1}, j2:{j2}, j3:{j3}, j4:{j4}")
                print(f"tau:{TAU}, k1:{k1}, k2:{k2}, k3:{k3}, k4:{k4}")
                
                # max_vel=0.99 #Keep speeds below the speed of light. 

                dv_nums[i]+=(1/6)*(j1+(2*j2)+(2*j3)+j4)
                # dv_nums[i]=min(dv_nums[i],max_vel)
                v_nums[i]+=(1/6)*(k1+(2*k2)+(2*k3)+k4)

                print(f"Velocity at tau={TAU} with respect to the coordinate: {i}: {dv_nums[i]}")
                print(f"Position at tau={TAU} with respect to the coordinate: {i}: {v_nums[i]}")
                TAU+=h
            print(f"Angular Momentum: {(R**2)*DPHI}")
            #Set The Initial Condition Again To Compute the Velocity of the Body
            CH_Symbols_To_Use.clear()  
            T=self.t_num
            R=new_radius
            THETA=self.theta_num
            PHI=self.phi_num


#Main Function
def main(): 
    # Initial Conditions For Circular Orbit
    t=0
    M=6.3e10**19 #Mass Of the Object
    r=500*M #Start position
    phi=0
    theta=np.radians(math.pi/2)
   

    T=Motion(t,r,theta,phi,M)
    T.GeodesicEquations()

main()


  