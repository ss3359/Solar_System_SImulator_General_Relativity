//
//  GIHON.cpp
//  RIVERS_XCODE
//
//  Created by Owner on 8/27/25.
//

#include <cmath>
#include <vector>
#include <thread>
#include <iostream>
#include <random>
#include <memory>
#include "GIHON.hpp"

using namespace std;


vector<double> operator-(vector<double>a, vector<double> b){
    vector<double> result(a.size());
    for(int i=0; i<a.size(); i++){
        result[i]= a[i]-b[i];
    }
    return result;
}

vector<double> operator*(double c, vector<double> v){
    vector<double> result(v.size());
    for(int i=0; i<v.size(); i++){
        result[i]= c*v[i];
    }
    return result;
}

double norm(vector<double> r){
    
    double sum=0.0;
    for(int i=0; i<r.size(); i++){
        sum+=r[i]*r[i];
    }
    return sqrt(sum);
}
vector<double> operator+(vector<double>a, vector<double> b){
    vector<double> result(a.size());
    for(int i=0; i<a.size(); i++){
        result[i]= a[i]+b[i];
    }
    return result;
}

vector<double> operator +=(vector<double>& a, const vector<double>& b){
    for(size_t i=0; i<a.size(); i++)
        a[i]+=b[i];
    
    return a;
}


Body::Body(vector<double> r0, vector<double> v0, double m){
    r=r0;
    v=v0;
    mass=m;
}

vector<double> Equation::dv_1(vector<Body> b,vector<double>K){
    vector<double>sum=(b[1].mass/pow(norm(b[0].r-b[1].r),3))*(b[0].r-b[1].r)
    +(b[2].mass/pow(norm(b[0].r-b[2].r),3))*(b[0].r-b[2].r);
    return G*sum;
}
vector<double>  Equation::dv_2(vector<Body> b,vector<double>K){
    vector<double>sum=(b[0].mass/pow(norm(b[1].r-b[0].r),3))*(b[1].r-b[0].r)
    +(b[2].mass/pow(norm(b[1].r-b[2].r),3))*(b[1].r-b[2].r);
    return G*sum;
}
vector<double>  Equation::dv_3(vector<Body> b,vector<double>K){
    vector<double>sum=(b[0].mass/pow(norm(b[2].r-b[0].r),3))*(b[2].r-b[0].r)
    +(b[1].mass/pow(norm(b[2].r-b[1].r),3))*(b[2].r-b[1].r);
    return G*sum;
}

vector<double> Equation:: dr_1(vector<double> J){
    return J;
}
vector<double> Equation:: dr_2(vector<double> J){
    return J;
}
vector<double> Equation:: dr_3(vector<double> J){
    return J;
    
}

void PrintVector(vector<double> d){
    for(int i=0; i<d.size(); i++){
        if(i==0)
            cout<<"<"<<d[i]<<",";
        else if(i==d.size()-1)
            cout<<d[i]<<">\n";
        else
            cout<<d[i]<<",";
    }
}

void RK45_Method(vector<Body> b){
    Equation e;
    int iter=10;
    
    double h=0.01; // Time Step in RK45 method

    vector<double> j11,j12,j13,j14;
    vector<double> k11,k12,k13,k14;
    
    vector<double> j21,j22,j23,j24;
    vector<double> k21,k22,k23,k24;
    
    vector<double> j31,j32, j33, j34;
    vector<double> k31,k32,k33,k34;
    
    for(int i=0; i<iter; i++){
        j11=e.dr_1(b[0].v);
        k11=e.dv_1(b, b[0].r);
        
        j21=e.dr_1(b[1].v);
        k21=e.dv_1(b, b[1].r);
        
        j31=e.dr_1(b[2].v);
        k31=e.dv_1(b, b[2].r);
        
        
    
        vector<double> step_size_a11=b[0].r+(0.5)*j11;
        vector<double> step_size_b11=b[0].v+(0.5)*k11;
        
        vector<double> step_size_a21=b[1].r+(0.5)*j21;
        vector<double> step_size_b21=b[2].v+(0.5)*k21;
        
        vector<double> step_size_a31=b[2].r+(0.5)*j31;
        vector<double> step_size_b31=b[2].v+(0.5)*k31;
            
        j12=e.dr_1(step_size_b11);
        k12=e.dv_2(b, step_size_a11);
        
        j22=e.dr_1(step_size_b21);
        k22=e.dv_1(b, step_size_a21);
        
        j32=e.dr_1(step_size_b31);
        k32=e.dv_1(b, step_size_a31);
        
        vector<double> step_size_a12=b[0].r+(0.5)*j12;
        vector<double> step_size_b12=b[0].v+(0.5)*k12;
        
        vector<double> step_size_a22=b[1].r+(0.5)*j22;
        vector<double> step_size_b22=b[1].v+(0.5)*k22;
        
        vector<double> step_size_a32=b[2].r+(0.5)*j32;
        vector<double> step_size_b32=b[2].v+(0.5)*k32;
        
        
        j13=e.dr_1(step_size_b12);
        k13=e.dv_2(b, step_size_a12);
        
        j23=e.dr_1(step_size_b22);
        k23=e.dv_1(b, step_size_a22);
        
        j33=e.dr_1(step_size_b32);
        k33=e.dv_1(b, step_size_a32);
        
        vector<double> step_size_a13=b[0].r+j13;
        vector<double> step_size_b13=b[0].v+k13;
        
        vector<double> step_size_a23=b[1].r+j23;
        vector<double> step_size_b23=b[1].v+k23;
        
        vector<double> step_size_a33=b[2].r+j33;
        vector<double> step_size_b33=b[2].v+k33;
        
        j14=e.dr_1(step_size_b13);
        k14=e.dv_2(b, step_size_a13);
        
        j24=e.dr_1(step_size_b23);
        k24=e.dv_2(b, step_size_a23);
        
        j34=e.dr_1(step_size_b33);
        k34=e.dv_2(b, step_size_a33);
        
        b[0].v=(h/6.0)*(k11+(3.0*k12)+(3.0*k13)+k14);
        b[0].r=(h/6.0)*(j11+(3.0*j12)+(3.0*j13)+j14);
        
        b[1].v=(h/6.0)*(k21+(3.0*k22)+(3.0*k23)+k24);
        b[1].r=(h/6.0)*(j21+(3.0*j22)+(3.0*j23)+j24);
        
        b[2].v=(h/6.0)*(k31+(3.0*k32)+(3.0*k33)+k34);
        b[2].r=(h/6.0)*(j31+(3.0*j32)+(3.0*j33)+j34);
        
        cout<<"Veclocity for Body 1"<<endl;
        PrintVector(b[0].v);
        cout<<"Velocity for Body 2"<<endl;
        PrintVector(b[1].v);
        cout<<"Velocity for Body 3"<<endl;
        PrintVector(b[2].v);
    }
    
}


