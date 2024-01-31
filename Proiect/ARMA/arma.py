###############################################################################

import numpy as np
import matplotlib.pyplot as plt

#====================================================
# generarea unei serii de timp cu o anomalie
#====================================================
def ser(length,caz=0):
    time=np.arange(length)
    trend=time**4/8/length**3 if caz else (time-length/2)**4*2/length**3
    noise=np.random.normal(0, 1, length)
    time_series=trend+noise
    return time_series # serie cu trend
n=100 # lungimea seriei
v=ser(n)
for i in range(50,70):
    v[i]+=3+min(i-50,70-i)/5 # produce anomalii

#%%

#================================================
# determinarea trendului
#================================================
def dt(serie,d,t=None): # d = gradul polinomului
    if len(np.shape(t))==0:
        t=np.arange(0,len(serie))
    A=np.vander(t,d+1,True).astype(float)
    A1=np.transpose(A)
    c=np.linalg.solve(A1@A,A1@serie)
    return c,sum(c[i]*t**i for i in range(d+1))

###############################################################################

#%%==========================================
#=================
# Optiuni pentru trend: grad 2 / 3 / 4 / 5
#=================
#%%==========================================

fig,ax=plt.subplots(2,2,num='Metode statistice: praguri hardcodate',figsize=(20,20),clear=True)
ax[0][0].plot(v,label='Serie initiala')
ax[0][0].legend(loc='best')

# metoda z-score (trend de grad 2 / 3 / 4 / 5)

anm=np.zeros(n)
grade=5
for d in range(2,grade+1):
    c,trend=dt(v,d)
    r=v-trend
    m=np.sum(r)/n
    sd=np.sqrt(sum((r-m)**2)/n)
    temp=np.where(abs(r-m)>=sd/4,r-m,0)
    sd=np.sqrt(sum(temp**2)/np.count_nonzero(temp))
    z=(r-m)/sd
    anm+=np.where(abs(z)>=1.7,d,0)
M=np.max(anm)
ax[0][1].plot(v,label='serie',color='b')
x,y=[],[]
for i in range(n):
    if anm[i]>=0.7*M:
        x.append(i)
        y.append(v[i])
ax[0][1].plot(x,y,label='anomalii (z-score)',linestyle='none',marker='o',markerfacecolor='r')
ax[0][1].legend(loc='best')

# metoda medie mobila (trend de grad 2 / 3 / 4 / 5)

anm=np.zeros(n)
ws=n//5
step=max(1,n//30) # cel mult 30 de pasi; altfel ar putea fi setat step=1
grade=5
for d in range(2,grade+1):
    c,trend=dt(v,d)
    r=(v-trend)
    r/=max(abs(r))
    anm0=np.zeros(n)
    for i in range(0,n-ws,step): # poate neglija ultimele intrari ale seriei
        m=sum(r[i:i+ws])/ws
        anm0[i:i+ws]+=r[i:i+ws]-m
    for i in range(n-ws-1,-1,-step): # acopera ultimele intrari ale seriei (dar le poate neglija pe primele)
        m=sum(r[i:i+ws])/ws
        anm0[i:i+ws]+=r[i:i+ws]-m
    anm+=np.where(abs(anm0)>=(ws+step)//step,d,0)
M=np.max(anm)
ax[1][0].plot(v,label='serie',color='b')
x,y=[],[]
for i in range(n):
    if anm[i]>=0.8*M:
        x.append(i)
        y.append(v[i])
ax[1][0].plot(x,y,label='anomalii (medie mobila)',linestyle='none',marker='o',markerfacecolor='r')
ax[1][0].legend(loc='best')

# metoda de deviatie medie absoluta (trend de grad 2 / 3 / 4 / 5)

anm=np.zeros(n)
grade=5
for d in range(2,grade+1):
    c,trend=dt(v,d)
    r=v-trend
    m=np.sort(r)[n//2]
    sd=sum(abs(r-m))/n
    temp=np.where(abs(r-m)>=sd/4,r-m,0)
    sd=sum(abs(temp))/np.count_nonzero(temp)
    z=(r-m)/sd
    anm+=np.where(abs(z)>=2,d,0)
M=np.max(anm)
ax[1][1].plot(v,label='serie',color='b')
x,y=[],[]
for i in range(n):
    if anm[i]>=0.7*M:
        x.append(i)
        y.append(v[i])
ax[1][1].plot(x,y,label='anomalii (deviatie absoluta)',linestyle='none',marker='o',markerfacecolor='r')
ax[1][1].legend(loc='best')

#%%=====================================================
#=====================
# Trend de grad 5: cautarea automata a pragurilor
#=====================
#%%=====================================================

# folosirea metodei procentuale cu procent aproximat de o metoda fourier

# calculare si eliminare trend
c,trend=dt(v,5)
r=v-trend
# calculare procent
z=abs(np.fft.fft(r))
s=sum(z[3:6]) # frecventele 3, 4, 5 pot indica anomalii
# nu cautam anomalii in frecvente mai mari decat 5 pentru ca aceste frecvente reprezinta
# sezonalitatea si zgomotul seriei
i=z[1:].argmax()+1 # frecventa cu cea mai mare amplitudine probabil nu reprezinta anomalii
if 3<=i and i<6:
    s-=z[i]
procent=s/sum(z)

def zScore(r,n,prag):
    m=np.sum(r)/n
    sd=np.sqrt(sum((r-m)**2)/n)
    temp=np.where(abs(r-m)>=sd/4,r-m,0)
    sd=np.sqrt(sum(temp**2)/np.count_nonzero(temp))
    z=(r-m)/sd
    x=[]
    for i in range(n):
        if abs(z[i])>=prag:
            x.append(i)
    return x

def medieMobila(r,n,prag):
    ws=n//5
    step=max(1,n//30)
    anm0=np.zeros(n)
    for i in range(0,n-ws,step): # poate neglija ultimele intrari ale seriei
        m=sum(r[i:i+ws])/ws
        anm0[i:i+ws]+=r[i:i+ws]-m
    for i in range(n-ws-1,-1,-step): # acopera ultimele intrari ale seriei (dar le poate neglija pe primele)
        m=sum(r[i:i+ws])/ws
        anm0[i:i+ws]+=r[i:i+ws]-m
    x=[]
    for i in range(n):
        if abs(anm0[i])>=(ws+step)//step*prag:
            x.append(i)
    return x

def devMedAbs(r,n,prag):
    m=np.sort(r)[n//2]
    sd=sum(abs(r-m))/n
    temp=np.where(abs(r-m)>=sd/4,r-m,0)
    sd=sum(abs(temp))/np.count_nonzero(temp)
    z=(r-m)/sd
    x=[]
    for i in range(n):
        if abs(z[i])>=prag:
            x.append(i)
    return x

# Pentru cele 3 metode procent(prag) este o functie descrescatoare

def metodaProcentuala(metoda,r,procent):
    n=len(r)
    pmin=1.0 # Pragul minim este 1
    pmax=2.0
    while len(metoda(r,n,pmax))/n>procent:
        pmax*=2
    prag=(pmin+pmax)/2
    while prag!=pmin and prag!=pmax:
        x=metoda(r,n,prag)
        if len(x)/n>procent:
            pmin=prag
        elif len(x)/n<procent:
            pmax=prag
        else:break
        prag=(pmin+pmax)/2
    return np.array(x)

fig,ax=plt.subplots(2,2,num='Metode statistice: praguri cautate',figsize=(20,20),clear=True)
ax[0][0].plot(v,label='Serie initiala')
ax[0][0].legend(loc='best')
i=metodaProcentuala(zScore,r,procent)
ax[0][1].plot(v,label='serie',color='b')
ax[0][1].plot(i,v[i],label='anomalii (z-score)',linestyle='none',marker='o',markerfacecolor='r')
ax[0][1].legend(loc='best')
i=metodaProcentuala(medieMobila,r,procent)
ax[1][0].plot(v,label='serie',color='b')
ax[1][0].plot(i,v[i],label='anomalii (medie mobila)',linestyle='none',marker='o',markerfacecolor='r')
ax[1][0].legend(loc='best')
i=metodaProcentuala(devMedAbs,r,procent)
ax[1][1].plot(v,label='serie',color='b')
ax[1][1].plot(i,v[i],label='anomalii (deviatie absoluta)',linestyle='none',marker='o',markerfacecolor='r')
ax[1][1].legend(loc='best')



#%%=========================================================================================
#========================================TESTARE===========================================#
#%%=========================================================================================



#%%===================================================
#===================
# Analiza setului de date portofolio_data.xls
#===================
#%%===================================================

import pandas as pd
import datetime

l=pd.read_csv('portfolio_data.xls')
ser=np.array(l).T
l=list(l.columns)
t=list(map(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y'),ser[0]))
t=np.array(list(map(lambda x: (x-t[0]).days,t)))

fig,ax=plt.subplots(2,2,num='z-score pe portfolio_data.xls',figsize=(20,20),clear=True)
for j in range(4):
    v=ser[j+1].astype(float)
    c,trend=dt(v,5,t)
    r=v-trend
    z=abs(np.fft.fft(r))
    s=sum(z[3:6])
    i=z[1:].argmax()+1
    if 3<=i and i<6:
        s-=z[i]
    procent=s/sum(z)
    i=metodaProcentuala(zScore,r,procent)
    ax[j//2][j%2].plot(t,v,label=f'serie {l[j+1]}',color='b')
    ax[j//2][j%2].plot(t[i],v[i],label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
    ax[j//2][j%2].legend(loc='best')
fig.tight_layout()

fig,ax=plt.subplots(2,2,num='medie mobila pe portfolio_data.xls',figsize=(20,20),clear=True)
for j in range(4):
    v=ser[j+1].astype(float)
    c,trend=dt(v,5,t)
    r=v-trend
    z=abs(np.fft.fft(r))
    s=sum(z[3:6])
    i=z[1:].argmax()+1
    if 3<=i and i<6:
        s-=z[i]
    procent=s/sum(z)
    i=metodaProcentuala(medieMobila,r,procent)
    ax[j//2][j%2].plot(t,v,label=f'serie {l[j+1]}',color='b')
    ax[j//2][j%2].plot(t[i],v[i],label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
    ax[j//2][j%2].legend(loc='best')
fig.tight_layout()

fig,ax=plt.subplots(2,2,num='deviatie absoluta pe portfolio_data.xls',figsize=(20,20),clear=True)
for j in range(4):
    v=ser[j+1].astype(float)
    c,trend=dt(v,5,t)
    r=v-trend
    z=abs(np.fft.fft(r))
    s=sum(z[3:6])
    i=z[1:].argmax()+1
    if 3<=i and i<6:
        s-=z[i]
    procent=s/sum(z)
    i=metodaProcentuala(devMedAbs,r,procent)
    ax[j//2][j%2].plot(t,v,label=f'serie {l[j+1]}',color='b')
    ax[j//2][j%2].plot(t[i],v[i],label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
    ax[j//2][j%2].legend(loc='best')
fig.tight_layout()

#%%===================================================
#===================
# Analiza seturilor de date de pe yahoo finance
# AAPL.csv (Apple)
# AMD.csv (AMD)
# BTC-USD.csv (bitcoin)
# GOOG.csv (Google)
# MSFT.csv (Microsoft)
# NVDA.csv (Nvidia)
#===================
#%%===================================================

def gen(file,met):
    l=pd.read_csv(file)
    ser=np.array(l).T
    l=list(l.columns)
    t=list(map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'),ser[0]))
    t=np.array(list(map(lambda x: (x-t[0]).days,t)))
    v=ser[4].astype(float) # "Close price"
    c,trend=dt(v,5,t) # Daca se doreste sa fie considerate anomalii si deviatiile
    # mai line, se foloseste grad mai mic pentru trend (2 / 3 / 4 in loc de 5)
    # pentru aceasta exemplificare, gradul ramane 5.
    r=v-trend
    z=abs(np.fft.fft(r))
    s=sum(z[3:6])
    i=z[1:].argmax()+1
    if 3<=i and i<6:
        s-=z[i]
    procent=s/sum(z)
    i=metodaProcentuala(met,r,procent)
    return t,v,i

met=[zScore,medieMobila,devMedAbs]
metode=['z-score','medie mobila','deviatie absoluta']
surse=['AAPL.csv','AMD.csv','BTC-USD.csv','GOOG.csv','MSFT.csv','NVDA.csv']
for p in range(3):
    fig,ax=plt.subplots(3,2,num=metode[p],figsize=(20,20),clear=True)
    for j in range(6):
        t,v,i=gen(surse[j],met[p])
        ax[j//2][j%2].plot(t,v,label='serie',color='b')
        ax[j//2][j%2].plot(t[i],v[i],label='anomalii',linestyle='none',marker='o',markerfacecolor='r')
        ax[j//2][j%2].legend(loc='best')
        ax[j//2][j%2].set_title(surse[j])
    fig.tight_layout()

#%%=====================================================
#=================
# ARMA
#=================
#%%=====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

n=100 # valoare mica si serie generata pentru ca ARIMA ruleaza greu
v=np.cumsum(np.random.randn(n))
p=n//10
qMax=min(5,p)
prag=1
pred=np.zeros(n)
for start in range(2*p,n): # pentru ultima valoare a seriei range(n-p,n)
    numar=0
    for q in range(1,qMax+1):
        try:
            model=ARIMA(v[:start],order=(p,0,q)) # 0 diferentieri
            model=model.fit()
            pred[start]+=model.forecast(steps=1)[0]
            print(start,q)
            numar+=1
        except:
            pass
    pred[start]/=numar if numar else pred[start-1]
s=sum(abs(pred-v))/(n-2*p) # pentru ultima valoare a seriei s=sum(abs(pred-v))/p
anm=[] # pentru ultima valoare a seriei se verifica  abs(pred[-1]-v[-1])>s*prag
for i in range(2*p,n):
    if abs(pred[i]-v[i])>s*prag:
        anm.append(i)
plt.figure()
plt.plot(v)
plt.plot(anm,v[anm],linestyle='none',marker='o')