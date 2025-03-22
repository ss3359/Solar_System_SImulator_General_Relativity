import math 
import random 
import numpy as np
import pandas as pd 
from mpmath import mp
from sympy import symbols,diff, Matrix
from sympy import re
from sympy import sqrt,acos,cos,sin,atan,tan
from sympy import Inverse,DotProduct, simplify
from sympy import Subs

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
c=3e8 # speed of light, (m/s)
mp.dps=50 # Set the desired percision (50 decimal places)
epsilon =1e-6 #Small Buffer to prevent division by near zero values
max_dr=1e-3
COLORS=["orange","green", "blue","red", "indigo", "purple","olive","cyan"]
iterations=10
Mass_Of_Sun=1.989e30 #Mass of the sun in kg


#Miscellaneous Functions To Use
def clamp(value,low,high):
    if(math.isnan(value)): 
        return low
    else:
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
    def __init__(self,t,r,theta,phi,NAME):
        self.tau=0
        self.t_num=t
        self.r_num=r
        self.theta_num=theta
        self.phi_num=phi
        self.m_num=Mass_Of_Sun
        self.name=NAME

        self.dt_discriminant=max(1-((2*G*self.m_num)/(self.r_num*(c**2))),1e-6)
        self.dt=1/sqrt(self.dt_discriminant)
        
        self.dtheta=0
        self.dphi = sqrt(G*self.m_num/self.r_num**3) #Orbital Velocity in spacetime
        self.a=self.r_num
        v_total=np.sqrt(G*Mass_Of_Sun/self.r_num*(2-(self.r_num/self.a)))
        self.dr=np.sqrt(v_total**2-(G*Mass_Of_Sun/self.r_num))
      
        self.R_NUMS=[self.t_num, self.r_num, self.theta_num,self.phi_num]
        print(f"Initial Motion Of Planet:{self.name}")
        print(f"t={self.t_num}, r={self.r_num}, θ={self.theta_num},φ={self.phi_num}")
        print(f"Mass={self.m_num}, dt={self.dt},dr={self.dr},dθ={self.dtheta}, dφ={self.dphi}")

    def R(self): 
        return [self.t,self.r,self.theta, self.phi]
      

    def SchwartzchildMETRIC(self):
        return Matrix([[-(1-(2*G*Mass_Of_Sun/self.r)),0,0,0],
                       [0, 1/(1-(2*G*Mass_Of_Sun/self.r)),0,0],
                       [0,0,self.r**2,0], 
                       [0,0,0,(self.r*sin(self.theta))**2]])
    
    def Christoffell(self):
        v=self.R()
        g=self.SchwartzchildMETRIC()
        g_inv=Matrix([[1/(1-(2*G*Mass_Of_Sun/self.r)),0,0,0],
                       [0, -(1-2*G*Mass_Of_Sun/self.r),0,0],
                       [0,0,1/self.r**2,0], 
                       [0,0,0,1/(self.r*sin(self.theta))**2]])
        Γ_array=[[["" for _ in range(4)] for _ in range(4)] for _ in range(4)]

    
        #The Algorithm to compute for the Christoffell Symbols
        for mu in range(4): 
            for nu in range(4): 
                for sigma in range(4): 
                    Γ_mu_nu_sigma=0   
                
                    for l in range(4):
                        Γ_mu_nu_sigma+=(1/2)*g_inv[sigma,l]*(diff(g[l,mu],v[nu])+diff(g[l,nu],v[mu])-diff(g[mu,nu],v[l]))
                    Γ_array[mu][nu][sigma]=simplify(Γ_mu_nu_sigma) 
        return Γ_array


    def GeodesicEquations(self): 
        t=symbols('t')
        v=[self.t,self.r,self.theta,self.phi]


        #Initial Conditions We Need
        T=self.t_num
        R=self.r_num

        THETA=self.theta_num
        PHI=self.phi_num

        DT=self.dt
        DR=self.dr
        DTHETA=self.dtheta
        DPHI=self.dphi
        TAU=0
        v_nums=[T,R,THETA,PHI]
        dv_nums=[DT,DR,DTHETA,DPHI]
       
        h=0.005 #time step.
        CH_Symbols_To_Use=[]
        low=-1e-5
        high=1e5
        

        # max_vel=0.99 #Keep speeds below the speed of light. 
        max_vel=0.99

        # Constants for RK Method      
        POSITION_VECTORS=[]
        VELOCITY_VECTORS=[]
        CH_Symbols=self.Christoffell()
        for iter in range(iterations): 

            j1=0
            j2=0
            j3=0
            j4=0

            k1=0
            k2=0
            k3=0
            k4=0 
            for i in range(4):
                sigma=i
                
                #RK4 Method for the First and Second Linear ODE Componentwise
                #Position And Velocity Of The Body
                for m in range(4): 
                    for n in range(4):
                        CH_num=CH_Symbols[m][n][sigma].subs([(v[0],T),(v[1],R),(v[2],THETA),(v[3],PHI)])
                        CH_num=CH_num.evalf()
                        CH_num=clamp(CH_num,low,high)
                        
                        j1+=h*(CH_num)*(dv_nums[m])*(dv_nums[n])
                        
                k1=h*(dv_nums[sigma])
        
                S=AddAComponentStep(i,j1/2)
                for m in range(4): 
                    for n in range(4):  
                        CH_num=CH_Symbols[m][n][sigma].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num=CH_num.evalf()
                        CH_num=clamp(CH_num,low,high)
                        j2+=h*(CH_num)*(dv_nums[m]+(j1/2))*(dv_nums[n]+(j1/2))
                k2=h*(dv_nums[sigma]+(k1/2))

                
                S=AddAComponentStep(i,j2/2)
                for m in range(4): 
                    for n in range(4): 
                        CH_num=CH_Symbols[m][n][sigma].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num=CH_num.evalf()
                        CH_num=clamp(CH_num,low,high)
                        
                        j3+=h*(CH_num)*(dv_nums[m]+(j2/2))*(dv_nums[n]+(j2/2))
                k3=h*(dv_nums[sigma]+(k2/2))

                
                S=AddAComponentStep(i,j3/2)
                for m in range(4): 
                    for n in range(4):
                        CH_num=CH_Symbols[m][n][sigma].subs([(v[0],T+S[0]),(v[1],R+S[1]),(v[2],THETA+S[2]),(v[3],PHI+S[3])])
                        CH_num=CH_num.evalf()
                        CH_num=clamp(CH_num,low,high)
                        
                        j4+=h*(CH_num)*(dv_nums[m]+(j3/2))*(dv_nums[n]+(j3/2))
                k4=h*(dv_nums[sigma]+(k3/2))

                dv_nums[i]+=(1/6)*(j1+(2*j2)+(2*j3)+j4)
                if(abs(dv_nums[i]) >max_vel): 
                    dv_nums[i]=max_vel*(dv_nums[i]/abs(dv_nums[i]))
                v_nums[i]+=(1/6)*(k1+(2*k2)+(2*k3)+k4)
            # print(f"Angular Momentum: {(R**2)*DPHI}")
            # print(f"New Position Vector: {v_nums}")
            POSITION_VECTORS.append(v_nums[:])
            VELOCITY_VECTORS.append(dv_nums[:])
            TAU+=h
           
        return POSITION_VECTORS,VELOCITY_VECTORS

class Planet: 
    #The constructor of this class has the proper spacetime coordinates as 
    #Cartesian Coordinates.
    def __init__(self,t,x,y,z,color,name,mass): 
        
        self.t=t
        self.r=x  
        self.theta=y  
        self.phi=z  
        self.color=color
        self.name=name
        self.m=mass #Mass is in kgs

    

    def GetPositionAndVelocity(self): 
        m=Motion(self.t,self.r,self.theta,self.phi,self.name)
        Position_Vectors,Velocity_Vectors=m.GeodesicEquations()
        return Position_Vectors,Velocity_Vectors
    
    def PlotOrbit(self,points): 
        P,V=self.GetPositionAndVelocity()
        fig,ax=plt.subplots(figsize=(6,6))
        ax.set_xlim(-1.6,1.6)
        ax.set_ylim(-1.6,1.6)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        #Plot The Sun
        # ax.scatter(0,0,color="yellow", s= 200,label="Sun")
        # print(f"Position Vectors:{P}")
        # for p in P:
        #     ax.scatter(p[1]*math.cos(p[3]),p[1]*math.sin(p[3]),color=self.color,label=self.name)
    
#Main Function
def main(): 
    
    #Initial Conditions for the planets 
    #1 AU is equal to 1.496e11 m
    AU=1.496e11
    PLANETS={
        "Mercury":{
            "Position":[0,0.387*AU,np.pi/2,0],
            "Mass":3.30e23,
            "Color":COLORS[0]
        },
         "Venus":{
            "Position":[0,0.723*AU,np.pi/2,np.pi/4],
            "Mass":4.87e24,
             "Color":COLORS[1]
        },
         "Earth":{
            "Position":[0,1.000*AU,np.pi/2,np.pi/2],
            "Mass":5.97e24, 
             "Color":COLORS[2]
        },
        "Mars":{
            "Position":[0,1.524*AU,np.pi/2,3*np.pi/4],
            "Mass":6.42e23, 
             "Color":COLORS[3]
        },
        "Jupiter":{
            "Position":[0,5.204*AU,np.pi/2,np.pi],
             "Mass":1.90e27,
             "Color":COLORS[4]
        },
        "Saturn":{
            "Position":[0,9.582*AU,np.pi/2,5*np.pi/4],
            "Mass":5.68e26, 
             "Color":COLORS[5]

        },
        "Uranus":{
            "Position":[0,19.22*AU,np.pi/2,3*np.pi/2],
            "Mass":8.68e25, 
             "Color":COLORS[6]

        },
        "Neptune":{
            "Position":[0,30.05*AU,np.pi/2,7*np.pi/4],
            "Mass":1.02e26, 
            "Color":COLORS[7]
            
        }
        
    }


    for name, data in PLANETS.items():
        P_data=data.get("Position")
        Color=data.get("Color")
        Name=name
        # if(Name=="Uranus"):
        #     breakpoint()
        planet=Planet(P_data[0],P_data[1],P_data[2],P_data[3],Color,Name,data.get("Mass"))
        
        P_Vectors,V_Vectors=planet.GetPositionAndVelocity()
        print("Position Vectors: ", P_Vectors)
        planet.PlotOrbit(P_Vectors)



    

main()

'''

   # Initial Conditions For Circular Orbit
    t=0
    M=6.3e10**19 #Mass Of the Object
    r=500*M #Start position
    phi=0
    theta=np.radians(math.pi/2)
   

T=Motion(t,r,theta,phi,M)
    T.GeodesicEquations()

      #Spherical Coordinates
        self.r=sqrt(x**2+y**2+z**2)
        self.theta=0
        self.phi=np.radians(math.pi/2)
        # self.theta=np.arctan(y/x)
        # self.phi=np.arccos(self.z/self.r)

          if abs(CH_num4) > 1e5:
                            print(f"WARNING: Large Christoffel symbol at tau={TAU}: {CH_num4}")
                            CH_num4=0


                    #Christoffell Symbols in question  
                    # p1=diff(g[sigma,mu],v[mu])
                    # p2=diff(g[sigma,mu],v[nu])
                    # p3=diff(g[mu,nu],v[sigma])
                    # print(f"∂g_{sigma}{mu}/∂v{mu}={p1}") 
                    # print(f"∂g_{sigma}{mu}/∂v{nu}={p2}") 
                    # print(f"∂g_{mu}{nu}/∂v{sigma}={p3}")  


                      
    # def GetAppropriateChristoffellSymbol(self,mu,nu,sigma): 
    #     Γ_array=self.Christoffell()
    #     for m in range(4):
    #         for n in range(4): 
    #             for s in range(4): 
    #                 if(sigma==s):
    #                     return [Γ_array[m][n][s],m,n]
                        
              # print("Metric tensor", g)
        # print("SC Metric: ", g)
        # print(f"Determinant of g: {g.det()}")      
        # 
        #   # self.dr=max(-self.dphi**2*self.r_num/self.dt_discriminant,1e6)
        # if self.d_phi_denominator>0 else 0 # (GM/r)Circular orbit approximation
        # if self.dt_discriminant>0:
        #     self.dr=-self.dphi**2*self.r_num/self.dt_discriminant
        # else: 
        #     self.dr=0
         # if(abs(j1)>0.1):
                        #     h*=0.5
                        #     h=min(h, 1e-4)
      #Set The Initial Condition Again To Compute RK for various coordinates
            # CH_Symbols_To_Use.clear()  
            # T=self.t_num
            # R=self.r_num
            # THETA=self.theta_num
            # PHI=self.phi_num

             # print(f"Velocity at tau={TAU} with respect to the coordinate: {i}: {dv_nums[i]}")
                # print(f"Position at tau={TAU} with respect to the coordinate: {i}: {v_nums[i]}")


                  # print(f"tau:{TAU}, j1:{j1}, j2:{j2}, j3:{j3}, j4:{j4}")
                # print(f"tau:{TAU}, k1:{k1}, k2:{k2}, k3:{k3}, k4:{k4}")
     
'''


  