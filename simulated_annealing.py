# coding: cp1252
"""
Le probème est celui dit du "voyageur de commerce" pour lequel un voyageur de commerce doit se rendre dans une
série de ville et cherche à minimiser la distance totale parcourue.

Ici, on impose au voyageur de partir de Paris et d'y revenir.

Pour n villes à visiter, il y a donc n! parcours possibles. La fonction coût est donc définie comme la
distance totale parcourue par le voyageur de commerce. Attention de bien prendre en compte le départ et le
retour de et vers Paris.

Il s'agit de trouver l'ordre dans lequel le voyageur de commerce doit parcourir les villes
pour minimiser le trajet qu'il a à effecuter. Pour ce faire, on utilise deux méthodes :
  * une méthode "brute force" pour laquelle vous calculerez tous les trajets possibles et sélectionnerez le
    plus court
  * la méthode du recuit simulé
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
          * nom : nom de la ville, chaine de caractère
          * xx : coordonnée en x par rapport à Paris
          * yy : coordonnée en y par rapport à Paris
        """
        self.infos[nom]={'x':float(xx), 'y':float(yy)}

################################
##### FUNCTION DEFINITIONS #####
################################
def cost_function(city_list, city_dict):
    """
    Fonction coût à minimiser. Pour une liste de villes donnée en entrée, la fonction calcule la distance à
    parcourir pour rallier toutes ces villes une à une. Attention, la fonction prend en compte la première
    distance de Paris et la dernière distance vers Paris.

    Entrée :
      * city_list : liste ordonnée des villes à parcourir, liste python
      * city_dict : dictionnaire des villes contenant les informations parmettant de
                    calculer les distances à parcourir, instance de la classe cities

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
    Fonction associée à la méthode de recuit simulé permettant de calculer un nouveau trajet candidat.

    Entrée :
      * list_in : une liste non ordonnée des villes à visiter par le voyageur de commerce, liste python

    Sortie :
      * une liste "aléatoire" ordonnée des villes à visiter par le voyageur de commerce, liste python
    """
    return np.random.permutation(np.asarray(list_in).tolist())
    
def compute_Temp(h,k,ind,Temp):
    """
    Fonction associée à la méthode de recuit simulé. Permet de calculer la nouvelles valeur de
    température à la fin d'une itération (voir algorithme du cours).

    Entrée :
      * h>0 : paramètre du calcul, float python. Plus h est petit, plus l'algorithme risque de rester
              piéger dans un minimum local. Plus h est grand, plus longue est la convergence de
              l'algorithme
      * k : paramètre de l'algorithme, integer python
      * ind : itération courante de l'algorithme, integer python
      * Temp : température courante de l'algorithme

    Sortie : 
      * nouvelle valeur du paramètre k de l'algorithme, integer python
      * nouvelle valeur de température
    """
    while ind <= np.exp((k-1)*h) or ind>np.exp(k*h):
         k+=1
 
         Temp=1.0/k     
    return k,Temp 



##################
##### SCRIPT #####
##################
##### Paramètres #####
#***** Dictionnaire des villes *****
dico=cities()

#***** Liste non ordonnée des villes à parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orleans','Strasbourg','Metz']

###### Résolution du problème en force brute #####
print "\n ##### Resolution du probleme en force brute #####"


#***** Calcul de toutes les permutations possibles *****
t1=time.time()
import itertools
trajets=list(itertools.permutations(parcours))

print 'Nombre de trajets etudies : ',len(trajets)

#***** Calcul de la fonction coût pour chaque permutation *****
couts=np.zeros(len(trajets))
for j,k in enumerate(trajets):
   couts[j]=cost_function(k, dico)

t2=time.time()
jmin=np.argmin(couts)
print 'Trajet le plus court :' ,trajets[jmin],' km:',couts[jmin] 
print 'Temps de calcul : ',t2-t1

##### Résolution du problème par la méthode du recuit simulé #####
print "\n ##### Resolution du probleme par la methode du recuit simule #####"

#***** Paramètres du calcul *****
#----- Initialisation -----
candOld=compute_new_candidate(parcours)
coutOld=cost_function(candOld,dico)
#----- Paramètres de l'algorithme -----
itermax=1500
hpar=1.0
kpar=1
Temp=1.0/kpar
Temp_list=[Temp]
print 'dstart:',coutOld
#***** Algorithme de résolution *****
t1=time.time()
for ind in xrange(itermax):
    #----- Calcul d'un nouveau trajet candidat -----
    candNew=compute_new_candidate(parcours)
    #----- Calcul de la différence de coût entre l'ancien et le nouveau trajet -----
    coutNew=cost_function(candNew, dico)
    #----- Si le nouveau trajet candidat est plut cher, il peut quand même -----
    #----- être accepté avec une certaine probabilité -----
    if coutNew<=coutOld:
        candOld=candNew; coutOld=coutNew    
    else:
        deltaE=coutNew-coutOld
        if np.exp(-deltaE/Temp)>=np.random.random():
            candOld=candNew; coutOld=coutNew
#    print 'dist=',coutOld
    #----- Diminution de la température -----
    kpar,Temp= compute_Temp(hpar,kpar,ind+2,Temp)
    Temp_list.append( Temp  )
t2=time.time()
#***** Résultat *****


print 'Trajet le plus court :' ,candOld ,' d:',coutOld
print 'Temps de calcul : ',t2-t1


#----- Profil de température -----
plt.figure()
plt.plot(Temp_list)
plt.xlabel('$n$')
plt.ylabel('$T$')
plt.title(u'Profil de temperature')
plt.grid()

plt.show()

