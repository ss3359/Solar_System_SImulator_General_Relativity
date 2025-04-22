import math 
import numpy as np
import pandas as pd
import sympy as sp 
import scipy as sip

import matplotlib.pyplot as plt
from matplotlib import animation
from sympy import Matrix,sin,cos

#Overview: 
     #This will implement the Geodesic Equation. I will use the Runge Kutta of the fourth order
        # To update the position and velocities of the planets in the Solar System
        # First, here is the Geodesic Equation 
        # x(rho)'' = - CS^{rho}{mu}{nu}(x(mu)')(x(nu)'). 
        # This can be xpressed as a system of First Order Linear Differential Equations
        # 
        #  D1 = x(rho)
        #  D2 = x(rho)'    
        #
        # D1' = D2 [t,ρ,θ,φ]
        # D2' = - CS^{rho}{mu}{nu}(x(mu)')(x(nu)') [t,ρ,θ,φ]
     

#Constants
G=6.67430e-11 # Gravitational Constant (Nm^2/kg^2)
et=sp.Matrix([1,0,0,0])
ex=sp.Matrix([0,1,0,0])
ey=sp.Matrix([0,0,1,0])
ez=sp.Matrix([0,0,0,1]) 
h=0.05 # seconds

t,r,theta,phi=sp.symbols('t r theta phi')
a,  M=sp.symbols('a M')

class Kerr: 
    def __init__(self,t,r,theta,phi,dt,dr,dtheta,dphi,a,M):
        self.t=t
        self.x=sp.sqrt(r**2+a**2)*cos(phi)*sin(theta)
        self.y=sp.sqrt(r**2+a**2)*sin(phi)*sin(theta)
        self.z=r*cos(theta)
        self.a=a
        self.M=M
        self.dt=dt
        self.dr=dr
        self.dtheta=dtheta
        self.dphi=dphi
    def R(self): 
        return np.array([self.t,self.x,self.y,self.z])
    def derR(self): 
        return np.array([self.dt,self.dr,self.dtheta,self.dphi])
       
        


    
    def MetricTensor(self):
        Variables=[t,r,theta,phi]
        Coordinates=[self.t,self.x,self.y,self.z]
        Basis_Vecors=[et,ex,ey,ez]

        G=sp.Matrix.zeros(4,4)
        for i in range(len(Variables)):
            for j in range(len(Variables)):
                partial_i=[sp.diff(Coordinates[k],Variables[i])*Basis_Vecors[k] for k in range(4)]
                partial_j=[sp.diff(Coordinates[k],Variables[j])*Basis_Vecors[k] for k in range(4)]
                
                Ri=sp.Matrix.zeros(4,1)
                Rj=sp.Matrix.zeros(4,1)
                
                for item1,item2 in zip(partial_i,partial_j): 
                    Ri+=item1
                    Rj+=item2
                
                G[i,j]=Ri.dot(Rj)
        return G 
    
    def InverseMetricTensor(self,Gμν): 
        InvGμν=Matrix.inv(Gμν)
        return InvGμν

    def Christoffel(self):
        # For the connection coefficients, known as the Christoffell Symbols, we need to determine the components that 
        # determine the measured direction of change. 

        Variables=[t,r,theta,phi]
        
        Gμν=self.MetricTensor()
        InvGμν=self.InverseMetricTensor(Gμν)
        Chirstoffell_Symbols=[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        Chirstoffell_Functions=[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]

        for rho in range(4):
            for mu in range(4):
                for nu in range(4): 
                    Γ_mu_nu_rho=0
                    for sigma in range(4):
                        Γ_mu_nu_rho+=0.5*(InvGμν[rho,sigma])*(sp.diff(Gμν[nu,sigma],Variables[mu]) 
                         + sp.diff(Gμν[mu,sigma],Variables[nu])
                         -sp.diff(Gμν[mu,nu],Variables[sigma]))
                    Chirstoffell_Symbols[mu][nu][rho]=sp.simplify(Γ_mu_nu_rho)
                    
                    for rho in range(4): 
                        for mu in range(4): 
                            for nu in range(4): 
                                Chirstoffell_Functions[mu][nu][rho]=sp.lambdify((t,r,theta,phi),Chirstoffell_Symbols[mu][nu][rho],modules='numpy')
        return Chirstoffell_Functions
    
    def D1(self,V,rho,J):
        return V[rho]+J
        
        
    def GetNeededCS(self,rho): 
        CS=self.Christoffel()
        CS_We_Need=[[[0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for mu in range(4):
            for nu in range(4):
                for p in range(4): 
                    if(p==rho):
                        CS_We_Need[mu][nu][rho]=CS[mu][nu][rho]     
        return CS_We_Need
    
    def D2(self,rho,t_num,r_num,theta_num,phi_num,K):
        CS_We_Need=self.GetNeededCS(rho)
        FUNCTON=0
        for mu in range(4):
            for nu in range(4):
                Γ=CS_We_Need[mu][nu][rho]
                Γ_term=Γ(t_num,r_num,theta_num,phi_num)
                FUNCTON+=-(Γ_term*self.derR()[mu]*self.derR()[nu]+K)
        return FUNCTON
        
    
    def UpdateVelocityAndPosition(self,t_num,r_num,theta_num,phi_num,dt_num,dr_num,dtheta_num,dphi_num):
        iter=1000
        Variables=[t,r,theta,phi]
        P=np.array([t_num,r_num,theta_num,phi_num])
        V=np.array([dt_num,dr_num,dtheta_num,dphi_num])

        P_vectors=[]
        V_vectors=[]
        # print(f"Initial Position: {P}")
        # print(f"Initial Velocity: {V}")
        for _ in range(iter):
            J1=[h*self.D1(V,i,0) for i in range(4)]
            K1=[h*self.D2(i,P[0],P[1],P[2],P[3],0) for i in range(4)]

            J2=[h*self.D1(V,i,J1[i]/2) for i in range(4)]
            K2=[h*self.D2(i,P[0],P[1],P[2],P[3],K1[i]/2) for i in range(4)]

            J3=[h*self.D1(V,i,J2[i]/2) for i in range(4)]
            K3=[h*self.D2(i,P[0],P[1],P[2],P[3],K2[i]/2) for i in range(4)]

            J4=[h*self.D1(V,i,J3[i]) for i in range(4)]
            K4=[h*self.D2(i,P[0],P[1],P[2],P[3],K3[i]) for i in range(4)]
            
            for i in range(4):
                P[i]+=(1/6)*(J1[i]+(2*J2[i])+(2*J3[i])+J4[i])
                V[i]+=(1/6)*(K1[i]+(2*K2[i])+(2*K3[i])+K4[i]) 
            print()
            # print(f"New Position: {P}")
            # print(f"New Velocity: {V}")
            P_vectors.append(P.copy())
            V_vectors.append(V.copy())
        return P_vectors
    
def PlotOrbit(a,P_Vectors_Kerr):
    X=[]
    Y=[]
    Z=[]
    for i in range(len(P_Vectors_Kerr)):
        X.append(np.sqrt(P_Vectors_Kerr[i][1]**2+a**2)*np.cos(P_Vectors_Kerr[i][3])*np.sin(P_Vectors_Kerr[i][2]))
        Y.append(np.sqrt(P_Vectors_Kerr[i][1]**2+a**2)*np.sin(P_Vectors_Kerr[i][3])*np.sin(P_Vectors_Kerr[i][2]))
        Z.append(P_Vectors_Kerr[i][1]*np.cos(P_Vectors_Kerr[i][2]))

    fig,ax=plt.subplots(figsize=(6,6))
    ax.set_xlim(min(X)*1.1,max(X)*1.1)
    ax.set_ylim(min(Y)*1.1,max(Y)*1.1)
    ax.set_title("Solar System Using Boyer-Lindquist Coordinates and Kerr Metric")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    ax.grid(True)
    ax.set_aspect('equal')

    line,=ax.plot([],[],'b--',label='Orbit Earth')
    planet,=ax.plot([],[],'ro',label="Planet")
    sun=ax.plot(0,0,'yo',label='Central Mass')
    ax.legend()

    def init():
        line.set_data([],[])
        planet.set_data([],[])
        return line,planet
    
    def update(frame):
        line.set_data(X[:frame],Y[:frame])
        planet.set_data(X[frame-1],Y[frame-1])
        return line,planet
    
    #Create the animation
    ani=animation.FuncAnimation(fig,update,frames=len(X),init_func=init,blit=True,interval=30)

    ax.show()

def main():
    M=1.989e30 #kgs
    t_num= 0 #seconds
    r_num=1.496e11 # meters
    theta_num=math.pi/2
    phi_num=0 
    dt_num=1 # placeholder
    dr_num=0
    dtheta_num=0
    dphi_num=1.99e-7 # rad/s Earth's angular velocity 

    a=0 #schwartzchild Limit (no spin)
    K=Kerr(t,r,theta,phi,dt_num,dr_num,dtheta_num,dphi_num,a,M)

    #RK4 Iterations
    P_Vectors_Kerr=K.UpdateVelocityAndPosition(t_num,r_num,theta_num,phi_num,dt_num,dr_num,dtheta_num,dphi_num)

    #Plot The Orbit
    PlotOrbit(a,P_Vectors_Kerr)
    
main()



