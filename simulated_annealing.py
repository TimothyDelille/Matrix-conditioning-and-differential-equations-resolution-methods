# coding: cp1252
"""
Le prob�me est celui dit du "voyageur de commerce" pour lequel un voyageur de commerce doit se rendre dans une
s�rie de ville et cherche � minimiser la distance totale parcourue.

Ici, on impose au voyageur de partir de Paris et d'y revenir.

Pour n villes � visiter, il y a donc n! parcours possibles. La fonction co�t est donc d�finie comme la
distance totale parcourue par le voyageur de commerce. Attention de bien prendre en compte le d�part et le
retour de et vers Paris.

Le travail demand� consiste � trouver l'ordre dans lequel le voyageur de commerce doit parcourir les villes
pour minimiser le trajet qu'il a � effecuter. Pour ce faire, on utilise deux m�thodes :
  * une m�thode "brute force" pour laquelle vous calculerez tous les trajets possibles et s�lectionnerez le
    plus court
  * la m�thode du recuit simul�
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

#############################
##### CLASS DEFINITIONS #####
#############################
class cities():
    def __init__(self):
        #***** Construction du dictionnaire des villes *****
        self.infos=dict()

        self.infos['Lille']      ={'x':52.0, 'y':197.0}
        self.infos['Orleans']    ={'x':-33.0, 'y':-105.0}
        self.infos['Lyon']       ={'x':185.0, 'y':-343.0}
        self.infos['Paris']      ={'x':0.0, 'y':0.0}
        self.infos['Marseille']  ={'x':225.0, 'y':-617.0}
        self.infos['Strasbourg'] ={'x':403.0, 'y':-31.0}
        self.infos['Rennes']     ={'x':-300.0, 'y':-88.0}
        self.infos['Metz']       ={'x':285.0, 'y':30.0}
        self.infos['Bordeaux']   ={'x':-213.0, 'y':-448.0}
        self.infos['Perpignan']  ={'x':40.0, 'y':-688.0}
        self.infos['Cherbourg']  ={'x':-289.0, 'y':86.0}

    def ajout(self,nom,xx,yy):
        """
        Methode d'ajout d'une ville au dictionnaire.

        Entrees :
          * nom : nom de la ville, chaine de caract�re
          * xx : coordonn�e en x par rapport � Paris
          * yy : coordonn�e en y par rapport � Paris
        """
        self.infos[nom]={'x':float(xx), 'y':float(yy)}

################################
##### FUNCTION DEFINITIONS #####
################################
def cost_function(city_list, city_dict):
    """
    Fonction co�t � minimiser. Pour une liste de villes donn�e en entr�e, la fonction calcule la distance �
    parcourir pour rallier toutes ces villes une � une. Attention, la fonction prend en compte la premi�re
    distance de Paris et la derni�re distance vers Paris.

    Entr�e :
      * city_list : liste ordonn�e des villes � parcourir, liste python
      * city_dict : dictionnaire des villes contenant les informations parmettant de
                    calculer les distances � parcourir, instance de la classe cities

    Sortie :
      * la distance parcourue, float python
    """
    d=0.
    xP = city_dict.infos['Paris']['x']    
    yP = city_dict.infos['Paris']['y']    
    xlast=xP;ylast=yP
    for i in city_list:
       xnow = city_dict.infos[i]['x']
       ynow = city_dict.infos[i]['y']    
       d+=( (xlast-xnow)**2.+(ylast-ynow)**2.) ** 0.5
       xlast=xnow;ylast=ynow
    d+=( (xP-xnow)**2.+(yP-ynow)**2.) ** 0.5
 
    return d

def compute_new_candidate(list_in):
    """
    Fonction associ�e � la m�thode de recuit simul� permettant de calculer un nouveau trajet candidat.

    Entr�e :
      * list_in : une liste non ordonn�e des villes � visiter par le voyageur de commerce, liste python

    Sortie :
      * une liste "al�atoire" ordonn�e des villes � visiter par le voyageur de commerce, liste python
    """
    return np.random.permutation(np.asarray(list_in).tolist())
    
def compute_Temp(h,k,ind,Temp):
    """
    Fonction associ�e � la m�thode de recuit simul�. Permet de calculer la nouvelles valeur de
    temp�rature � la fin d'une it�ration (voir algorithme du cours).

    Entr�e :
      * h>0 : param�tre du calcul, float python. Plus h est petit, plus l'algorithme risque de rester
              pi�ger dans un minimum local. Plus h est grand, plus longue est la convergence de
              l'algorithme
      * k : param�tre de l'algorithme, integer python
      * ind : it�ration courante de l'algorithme, integer python
      * Temp : temp�rature courante de l'algorithme

    Sortie : 
      * nouvelle valeur du param�tre k de l'algorithme, integer python
      * nouvelle valeur de temp�rature
    """
#    print 'k:',k ,'   h:',h,' c1',np.exp((k-1)*h),' c2:',np.exp(k*h) ,'temp:' ,Temp ,' ind:' ,ind
#    plt.pause(0.01)
    while ind <= np.exp((k-1)*h) or ind>np.exp(k*h):
         k+=1
#         print k  
         Temp=1.0/k     
    return k,Temp 



##################
##### SCRIPT #####
##################
##### Param�tres #####
#***** Dictionnaire des villes *****
dico=cities()

#***** Liste non ordonn�e des villes � parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orleans','Strasbourg','Metz']

###### R�solution du probl�me en force brute #####
print "\n ##### Resolution du probleme en force brute #####"


#***** Calcul de toutes les permutations possibles *****
t1=time.time()
import itertools
trajets=list(itertools.permutations(parcours))

print 'Nombre de trajets etudies : ',len(trajets)

#***** Calcul de la fonction co�t pour chaque permutation *****
couts=np.zeros(len(trajets))
for j,k in enumerate(trajets):
   couts[j]=cost_function(k, dico)

t2=time.time()
jmin=np.argmin(couts)
print 'Trajet le plus court :' ,trajets[jmin],' km:',couts[jmin] 
print 'Temps de calcul : ',t2-t1

##### R�solution du probl�me par la m�thode du recuit simul� #####
print "\n ##### Resolution du probleme par la methode du recuit simule #####"

#***** Param�tres du calcul *****
#----- Initialisation -----
candOld=compute_new_candidate(parcours)
coutOld=cost_function(candOld,dico)
#----- Param�tres de l'algorithme -----
itermax=1500
hpar=1.0
kpar=1
Temp=1.0/kpar
Temp_list=[Temp]
print 'dstart:',coutOld
#***** Algorithme de r�solution *****
t1=time.time()
for ind in xrange(itermax):
    #----- Calcul d'un nouveau trajet candidat -----
    candNew=compute_new_candidate(parcours)
    #----- Calcul de la diff�rence de co�t entre l'ancien et le nouveau trajet -----
    coutNew=cost_function(candNew, dico)
    #----- Si le nouveau trajet candidat est plut cher, il peut quand m�me -----
    #----- �tre accept� avec une certaine probabilit� -----
    if coutNew<=coutOld:
        candOld=candNew; coutOld=coutNew    
    else:
        deltaE=coutNew-coutOld
        if np.exp(-deltaE/Temp)>=np.random.random():
            candOld=candNew; coutOld=coutNew
#    print 'dist=',coutOld
    #----- Diminution de la temp�rature -----
    kpar,Temp= compute_Temp(hpar,kpar,ind+2,Temp)
    Temp_list.append( Temp  )
t2=time.time()
#***** R�sultat *****


print 'Trajet le plus court :' ,candOld ,' d:',coutOld
print 'Temps de calcul : ',t2-t1


#----- Profil de temp�rature -----
plt.figure()
plt.plot(Temp_list)
plt.xlabel('$n$')
plt.ylabel('$T$')
plt.title(u'Profil de temperature')
plt.grid()

plt.show()

