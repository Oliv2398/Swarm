##### Projet NOIRE | 2020 - 2021
##### Trajectoires poule et poussins


### Importation des modules
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


#____________________Paramètres____________________

nb_poussin = 48 # nombre de poussins
X = 2 # nombre de periode (X=None pour une mission d'une annee terrestre)
precision = 500 # nombre de points sur une periode (min 100 pour faire le largage des 48 poussins sur une demi periode)


#_________________Options_graphiques_______________

plot2D = False # graphique 2D des trajectoires autour de la Lune
plot3D = False # graphique 3D des trajectoires autour de la Lune
dist_std = True # Distances relatives, ecart-type, moyenne et distance max des poussins a la poule en fonction du temps.

traj_ref_lune = False # Plot en 3D anime des trajectoires autour de la lune
animated = False # Plot en 3D anime des positions relatives des poussins et de la poule.
                 # Differents referentiels sont utilisables. (choix ligne 648)

# PS : pour plus de precision sur une fonction ecrire help(fonction_name) dans le terminal
#      apres avoir execute au moins une fois le programme


#____________________Constantes____________________
G = 6.67259e-11 # SI, constante de gravitation
dL_P = 5000e3 # m, distance Lune-poule
M_L = 7.3477e22 # kg, masse Lune
M_P = 1e3 # kg, masse poule
V_P = np.sqrt(G*M_L/dL_P) # vitesse poule : R10k:~700 m/s R5k:~990
period = np.sqrt(dL_P**3*4*np.pi**2/(G*M_L)) # T R10k:~24,9 h R5k:~8,8h
nb_dimension = 3


#______________Impulsion max poussins______________
C = 100e3 # contrainte de distance -> 100km
DV_max = 0.25*V_P*C/dL_P # impulsion maximale

term1 = DV_max * np.sqrt(1-(DV_max/(2*V_P))**2) # Dv * sqrt(1 - (Dv/2vm)^2)
term2 = DV_max**2/(2*V_P) # Dv^2/2vm


#____________________Fonctions____________________

### Gravitation
fL_Px = lambda x_t, y_t, z_t : -G*M_L*x_t/((x_t**2+y_t**2+z_t**2)**(3/2))
fL_Py = lambda x_t, y_t, z_t : -G*M_L*y_t/((x_t**2+y_t**2+z_t**2)**(3/2))
fL_Pz = lambda x_t, y_t, z_t : -G*M_L*z_t/((x_t**2+y_t**2+z_t**2)**(3/2))


### Algos resolution
# NB : _algo_verlet2 seul a bien propager les trajectoires avec l'impulsion

def _algo_euler1(x, v, F, i, dv):
    """
    Algorithme d'integration basique, Euler explicite.
    1er ordre, Perte d'energie constante.

    Parameters:
    - x : tab, positions du corps (x,y,z)
    - v : tab, vitesses du corps (vx, vy, vz)
    - F : tab, tableau des forces (Fx, Fy, Fz)
    - i : int, iteration actuelle
    - dv : tab, tableau des impulsions a donner

    Return:
    - x2 : tab, nouvelles positions du corps
    - v2 : tab, nouvelles vitesses du corps
    """
    x2 = np.zeros(nb_dimension)
    v2 = np.zeros(nb_dimension)
    for k in range(nb_dimension):
        x2[k] = x[i,k] + dt * v[i,k]
    for k in range(nb_dimension):
        v2[k] = v[i,k] + dt * F[k](*x2) + dv[k]
    return x2, v2

def _algo_euler2(x, v, F, i, dv):
    """
    Algorithme d'integration Euler symplectique.

    Parameters:
    - x : tab, positions du corps (x,y,z)
    - v : tab, vitesses du corps (vx, vy, vz)
    - F : tab, tableau des forces (Fx, Fy, Fz)
    - i : int, iteration actuelle
    - dv : tab, tableau des impulsions a donner

    Return:
    - x2 : tab, nouvelles positions du corps
    - v2 : tab, nouvelles vitesses du corps
    """
    x2 = np.zeros(nb_dimension)
    v2 = np.zeros(nb_dimension)
    for k in range(nb_dimension):
        x2[k] = x[i,k] + dt * v[i,k]
        x[i,k] += dt*v[i,k]
    for k in range(nb_dimension):
        v2[k] = v[i,k] + dt*F[k](*x[i]) + dv[k]
    return x2, v2

def _algo_verlet1(x, v, F, i, dv):
    """
    Algorithme d'integration verlet a 1 pas.
    2e ordre, conservation de l'energie.

    Parameters:
    - x : tab, positions du corps (x,y,z)
    - v : tab, vitesses du corps (vx, vy, vz)
    - F : tab, tableau des forces (Fx, Fy, Fz)
    - i : int, iteration actuelle
    - dv : tab, tableau des impulsions a donner

    Return:
    - x2 : tab, nouvelles positions du corps
    - v2 : tab, nouvelles vitesses du corps
    """
    x2 = np.zeros(nb_dimension)
    v2 = np.zeros(nb_dimension)
    for k in range(nb_dimension):
        x2[k] = x[i,k] + dt*v[i,k] + (dt**2/2)*F[k](*x[i])
    for k in range(nb_dimension):
        v2[k] = v[i,k] + (dt/2)*(F[k](*x[i]) + F[k](*x2)) + dv[k]
    return x2, v2

def _algo_verlet2(x, F, i, dv): #
    """
    Algorithme d'integration verlet a 2 pas (pas besoin de la vitesse ici, mais besoin des 2 premieres coordonnees).
    2e ordre, conservation de l'energie.

    Parameters:
    - x : tab, positions du corps (x,y,z)
    - F : tab, tableau des forces (Fx, Fy, Fz)
    - i : int, iteration actuelle
    - dv : tab, tableau des impulsions a donner

    Return:
    - x2 : tab, nouvelles positions du corps
    """
    x2 = np.zeros(nb_dimension)
    for k in range(nb_dimension):
        x2[k] = 2.0*x[i,k] - x[i-1,k] + dt**2*F[k](*x[i]) + dt*dv[k]
    return x2


### Integration, calcul des trajectoires, largage et propagation
def Trajectories(algo, verbose=True):
    """
    Fonction principale, calcule les trajectoires des poussins et de la poule avec des largages sur une demi période.

    Parameter:
    - algo : func, algorithme a utiliser pour effectuer l'integration

    Optional parameter:
    - verbose : bool, print l'avancement des calculs

    Return:
    - pos : tab, tableau de positions de la poule ; shape = (N, nb_dimension)
    - pos_nanos : tab, tableau de positions des poussins ; shape = (N, nb_poussin, nb_dimension)
    """

    assert algo in (_algo_euler1, _algo_euler2, _algo_verlet1, _algo_verlet2), "choisir un algo existant"

    pos = np.zeros((N, nb_dimension)) # poule
    vit = np.zeros((N, nb_dimension))
    pos_nanos = np.zeros((N,nb_poussin, nb_dimension)) # poussin
    vit_nanos = np.zeros((N,nb_poussin, nb_dimension))
    dVk = np.zeros((nb_poussin, nb_dimension)) # impulsions

    # conditions initiales de la poule et des poussins
    pos[0,0] = dL_P ; vit[0,1] = V_P
    for j in range(nb_poussin):
        pos_nanos[0,j,0] = dL_P ; vit_nanos[0,j,1] = V_P

    # instant des lancements, toujours sur une demi-periode
    t_lance = np.floor(np.arange(int(nb_poussin/2)) *period/nb_poussin/dt +1)

    # Forces
    F_L_P = [fL_Px, fL_Py, fL_Pz]

    # angle de rotation de lancement et numero du lancement
    theta=0 ; pack=0

    # Main loop
    for i in range(N-1) :

        # init pos+1, dans le cas de l'algo verlet2 uniquement
        if i==0 and algo==_algo_verlet2:
            v = [0,V_P,0]
            for j in range(nb_dimension):
                pos[1,j] = pos[0,j] + dt*v[j] + dt**2/2 * F_L_P[j](*pos[0])
                for k in range(nb_poussin):
                    pos_nanos[1,k,j] = pos_nanos[0,k,j] + dt*v[j] + dt**2/2 * F_L_P[j](*pos_nanos[0,k])
            continue

        # ________________Mother________________
        if algo!=_algo_verlet2:
            pos[i+1], vit[i+1] = algo(pos,vit,F_L_P,i,[0,0,0])
        else:
            pos[i+1] = algo(pos,F_L_P,i,[0,0,0])

        # _______________Nanos pos________________
        lance = np.zeros(nb_poussin)
        if i in t_lance: # lancements
            phi = np.arccos(pos[i,0]/np.sqrt(pos[i,0]**2 +pos[i,1]**2 +pos[i,2]**2 ))

            lance[pack*2]+=1 ; lance[pack*2+1]+=1 # seuls les 2 poussins concernés auront une impulsion non nulle
            if verbose:
                print("___LANCEMENT___ || Satellites [{:}] et [{:}]".format(pack*2+1,pack*2+2), end='\r')

            for j in range(0,nb_poussin,2): # Impulsions
                dVk[j,0] = -np.sin(theta)*np.cos(phi)*term1 + np.sin(phi)*term2
                dVk[j,1] = -np.sin(theta)*np.sin(phi)*term1 - np.cos(phi)*term2
                dVk[j,2] = -np.cos(theta)*term1

                dVk[j+1,0] = np.sin(theta)*np.cos(phi)*term1 + np.sin(phi)*term2
                dVk[j+1,1] = np.sin(theta)*np.sin(phi)*term1 - np.cos(phi)*term2
                dVk[j+1,2] = np.cos(theta)*term1

            for j in range(nb_poussin):
                # ajout d'une impulsion au 2 poussins concernés, propagation des autres
                if algo!=_algo_verlet2:
                    pos_nanos[i+1,j], vit_nanos[i+1,j] = algo(pos_nanos[:,j], vit_nanos[:,j], F_L_P, i, dVk[j]*lance[j])
                else:
                    pos_nanos[i+1,j] = algo(pos_nanos[:,j], F_L_P, i, dVk[j]*lance[j])

            theta += 2*np.pi/(nb_poussin/2)
            pack+=1

        else : # propagation
            for j in range(nb_poussin):
                if algo!=_algo_verlet2:
                    pos_nanos[i+1,j], vit_nanos[i+1,j] = algo(pos_nanos[:,j],vit_nanos[:,j],F_L_P,i,[0,0,0])
                else:
                    pos_nanos[i+1,j] = algo(pos_nanos[:,j], F_L_P, i, [0,0,0])

        if verbose and pack*2 >= nb_poussin :
            if X:
                print("---------- {} % ---------- [{}/{}] periode ----------".format(int(1+i/N*100), int(1+i*dt/period), X),end='\r')
            else:
                print("------ {} % ------ [{}/{}] periode ------ {} mois ------".format(int(1+i/N*100), int(1+i*dt/period), int(1+N*dt/period),int(1+i/N*12)), end='\r')

    return pos, pos_nanos

#_______________________________________________________________
#_____________________________ Plots ___________________________
#_______________________________________________________________

# Static Plots
def Plot2D(mother, nanos, legend=False):
    """
    Plot selon les axes X et Y des trajectoires.

    Parameters:
    - mother : tab, positions de la poule
    - nanos : tab, positions des poussins

    Optional parameter:
    - legend : bool, presence de la legende ou non dans le plot
    """
    plt.figure()
    for i in range(nb_poussin):
        plt.plot(*nanos.T[:,i][:2]/1e3,label='nanos '+str(i+1))
    plt.plot(*mother.T[:2]/1e3,'k',label='poule')
    plt.plot(0,0,'ok',label='Lune')
    if legend:
        plt.legend()
    plt.axis('equal')
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.show()

def Plot3D(mother, nanos, legend=False):
    """
    Plot en 3 dimensions des trajectoires.

    Parameters:
    - mother : tab, positions de la poule
    - nanos : tab, positions des poussins

    Optional parameter:
    - legend : bool, presence de la legende ou non dans le plot
    """
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.plot(*mother.T/1e3, 'k', label='trajectoire poule')
    ax.plot([0],[0],[0], 'ok', label='Lune')

    color=iter(plt.cm.rainbow(np.linspace(0,1,int(nb_poussin))))
    for i in range(nb_poussin):
        col=next(color)
        ax.plot(*nanos[::,i].T/1e3, c=col, label='nano '+str(i+1))

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.view_init(55,33)
    if legend:
        plt.legend()
    plt.show()

def Dist_std(mother, nanos, individual=True, std=False, interval=None):
    """
    Distances relatives, ecart-type, moyenne et distance max des poussins a la poule en fonction du temps.

    Parameters:
    - mother : tab, positions de la poule
    - nanos : tab, positions des poussins

    Optional parameters:
    - individual : bool, afficher ou non le graphique des distances relatives
    - std : bool, afficher ou non l'ecart-type, la moyenne et la distance max
    - interval : tuple, interval de periodes a afficher
    """

    mother_rg = mother.copy()
    nanos_rg = nanos.copy()
    len_tab = N
    interval_0 = 0 ; interval_1 = N
    if interval:
        interval_0 = int(interval[0]*period/dt)
        interval_1 = int(interval[1]*period/dt)
        inter = interval_1 - interval_0

        mother_rg = mother_rg[interval_0:interval_1]
        nanos_rg = nanos_rg[interval_0:interval_1]
        len_tab = inter

    rad = np.sqrt(mother_rg.T[0]**2+mother_rg.T[1]**2+mother_rg.T[2]**2)
    xm, ym, zm = mother_rg.T

    tab = np.zeros((nb_poussin, len_tab))
    for i in range(nb_poussin):
        xp, yp, zp = nanos_rg.T[:,i]
        nanos_rad = np.sqrt((xp-xm)**2+(yp-ym)**2+(zp-zm)**2)
        tab[i] = nanos_rad

    if individual:
        plt.figure(1)
        plt.title("Distances poussins-poule")
        color=iter(plt.cm.rainbow(np.linspace(0,1,int(nb_poussin))))
        for i in range(nb_poussin):
            col=next(color)
            plt.plot(range(interval_0,interval_1)*dt/period, tab[i]/1e3, c=col)
        plt.xlabel('phase orbitale')
        plt.ylabel('distance (km)')
        plt.show()

    if std:
        tab_std, tab_mean, tab_max = np.zeros((3,len_tab))
        for i in range(len_tab):
            tab_std[i] = np.std(tab[:,i])
            tab_mean[i] = np.mean(tab[:,i])
            tab_max[i] = np.max(tab[:,i])

        plt.figure(2)
        plt.title("Ecart-type, moyenne et distance max poussins-poule")
        plt.plot(range(interval_0,interval_1)*dt/period, tab_std/1e3, 'r', label = "std")
        plt.plot(range(interval_0,interval_1)*dt/period, tab_mean/1e3, 'g', label = "mean")
        plt.plot(range(interval_0,interval_1)*dt/period, tab_max/1e3, 'k', label = "max")
        plt.xlabel('phase orbitale')
        plt.ylabel('distance (km)')
        plt.legend()
        plt.show()


# Trajectoires animees
def _update_lines(num, dataLines, lines):
    """
    Dependance de la fonction -Traj_animated-, parametre de la fonction -animation.FuncAnimation-.

    Parameters : num (iteration actuelle), dataLines (tableau des positions), lines (objets matplotlib, plots)
    """
    for line, data in zip(lines, dataLines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines
def Traj_animated(mother, nanos, legend=False, save=False):
    """
    Plot en 3 dimensions anime des trajectoires autour de la lune.

    Parameters:
    - mother : tab, positions de la poule
    - nanos : tab, positions des poussins

    Optional parameters:
    - legend : bool, presence de la legende ou non dans le plot
    - save : bool, enregistrement au format mp4 de l'animation cree
    """

    mother /= 1e3 # echelle en km
    nanos /= 1e3

    if save: # precaution si l'option save est activee
        print("\n\n----------------")
        print("Enregistrement :")
        check=False
        while check==False:
            valid = str(input("\nVerifier qu'aucune video ne comporte le meme nom que celui de l'enregistrement en cours "+
                    "(si oui alors elle sera effacee)."+
                    "\nPoursuivre ? y/n\n"))

            if valid in ('n','N'):
                check=True
                return 0
            elif valid in ('y','Y'):
                check=True
                pass
            else:
                check=False

    tab = np.zeros((nb_poussin+1,nb_dimension,N)) # creation d'un tableau regroupant la poule ET les poussins
    tab[0] = mother.T
    for i in range(1,nb_poussin+1):
        for j in range(nb_dimension):
            tab[i,j] = nanos[:,i-1][:,j]

    # Creation de la figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.view_init(50,-130) # angle de vue initial
    ax.set_xlim3d([-1.1*dL_P/1e3, 1.1*dL_P/1e3]) # limites des axes
    ax.set_ylim3d([-1.1*dL_P/1e3, 1.1*dL_P/1e3])
    ax.set_zlim3d([-30, 30])
    ax.set_xlabel('X (km)') ; ax.set_ylabel('Y (km)') ; ax.set_zlabel('Z (km)')

    ax.plot([0],[0],[0], 'ok', markersize=15, label='Lune') # point au milieu pour la lune

    lines=[] ; labels = ["poule"]
    lines.append(ax.plot(*tab[0], c='k', label=labels[0])[0])

    color=iter(plt.cm.rainbow(np.linspace(0,1,int(nb_poussin))))
    for i, dat in enumerate(tab[1:]):
        col=next(color)
        labels.append("poussin "+str(i+1))
        lines.append(ax.plot(*dat, c=col, label=labels[i+1])[0])

    traj_ani = animation.FuncAnimation(fig, _update_lines, N, fargs=(tab, lines), interval=1, blit=False)

    if save:
        print("Saving... (peut prendre du temps en fonction des choix de N et du nombre de poussins)")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
        traj_ani.save('trajs.mp4', writer=writer)

    if legend:
        plt.legend()
    plt.show(block=False)

    mother *= 1e3 # retablissement de l'echelle en metres
    nanos *= 1e3


# Positions relatives des poussins et de la mere animees
def Relat_pos_animated(mother, nanos, ref='poule', ref_poule='tournant', save=False):
    """
    Plot en 3 dimensions anime des positions relatives des poussins et de la poule.
    Differents referentiels sont utilisables : -lune- ou -poule-.
    Si le referentiel choisi est celui de la poule, on peut choisir entre :
        un referentiel translate uniquement -r_poule-
        un referentiel translate et rotate -tournant-.

    Parameters:
    - mother : tab, positions de la poule
    - nanos : tab, positions des poussins

    Optional parameters:
    - ref : str, choisir entre le referentiel de la -poule- ou de la -lune-
    - ref_poule : str, (! si ref==poule !) choisir entre -r_poule- et -tournant-
    - save : bool, enregistrement au format mp4 de l'animation cree
    """

    assert ref in ('lune','poule'), "Choisir entre le referentiel 'lune' et 'poule'."
    assert ref_poule in ('r_poule','tournant'), "Choisir un referentiel de translation et rotation, 'tournant', ou de translation uniquement, 'r_poule'."

    if save:
        print("\n\n----------------")
        print("Enregistrement :")
        check=False
        while check==False:
            valid = str(input("\nVerifier qu'aucune video ne comporte le meme nom que celui de l'enregistrement en cours "+
                    "(si oui alors elle sera effacee)."+
                    "\nPoursuivre ? y/n\n"))

            if valid in ('n','N'):
                check=True
                return 0
            elif valid in ('y','Y'):
                check=True
                pass
            else:
                check=False

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xnanos = np.zeros((N,nb_poussin))
    ynanos = np.zeros((N,nb_poussin))
    znanos = np.zeros((N,nb_poussin))
    for i in range(N):
        xnanos[i], ynanos[i], znanos[i] = nanos[i].T/1e3

    if ref == 'poule':
        for i in range(N):
            if ref_poule=='r_poule': # ref translation uniquement
                xnanos[i] -= mother[i,0]/1e3
                ynanos[i] -= mother[i,1]/1e3
                znanos[i] -= mother[i,2]/1e3

            if ref_poule=='tournant': # ref translation et rotation
                xp, yp, zp = nanos[i].T
                xm, ym, zm = mother[i]

                xnanos[i] = (np.sqrt(xp**2+yp**2)*np.cos(np.arctan((yp*xm-xp*ym)/(xp*xm+yp*ym))) - dL_P)/1e3
                ynanos[i] = np.sqrt(xp**2+yp**2)*np.sin(np.arctan((yp*xm-xp*ym)/(xp*xm+yp*ym)))/1e3
                znanos[i] = zp/1e3

        ax.set_xlim3d([-100,100])
        ax.set_ylim3d([-100,100])
        ax.set_zlim3d([-50,50])
        ax.plot([0], [0], [0], "ok", markersize=5) # poule fixe au milieu

    if ref == 'lune':
        ax.set_xlim3d([-1.1*dL_P/1e3, 1.1*dL_P/1e3])
        ax.set_ylim3d([-1.1*dL_P/1e3, 1.1*dL_P/1e3])
        ax.set_zlim3d([-30, 30])

    def update_graph(num):
        """
        Parametre de la fonction -animation.FuncAnimation- ligne 546.

        Parameter : num (iteration actuelle)
        """
        graph._offsets3d = (xnanos[num], ynanos[num], znanos[num])
        title.set_text('Referentiel {} ; Periode = {:.2f}'.format(ref, num*dt/period))

    title = ax.set_title('')
    color = plt.cm.rainbow(np.linspace(0,1,int(nb_poussin)))
    graph = ax.scatter(xnanos[0], xnanos[1], xnanos[2],
        c=color, s=40, alpha=1, marker='.')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    ani = animation.FuncAnimation(fig, update_graph, N, interval=1, blit=False)

    if save:
        print("Saving... (peut prendre du temps en fonction des choix de N et du nombre de poussins)")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=100, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('relat_trajs.mp4', writer=writer)

    plt.show(block=False)


# Informations sur la mission, poussins perdus
def Outliers(precis=False):
    """
    Informations sur le nombre de poussins perdus au cours de la mission.
    (Sortant de la limite des 100km.)

    Optional parameter:
    - precis : bool, informations sur les poussins en dehors des 100km
    """

    # tableau des positions lors de la derniere periode
    end_N = nanos[int((N*dt/period-1)*period/dt):-1]
    end_M = mother[int((N*dt/period-1)*period/dt):-1]

    rad = np.sqrt(end_M.T[0]**2+end_M.T[1]**2+end_M.T[2]**2)
    xm, ym, zm = end_M.T
    tab_N = np.zeros((nb_poussin, len(end_M)))
    for i in range(nb_poussin):
        xp, yp, zp = end_N.T[:,i]
        nanos_rad = np.sqrt((xp-xm)**2+(yp-ym)**2+(zp-zm)**2)
        tab_N[i] = nanos_rad

    sup = 0 ; sup2 = 0
    theta = 0
    for i in range(nb_poussin): # fraction en dehors des 100km
        if (tab_N[i]>100e3).any() and not (tab_N[i]>100e3).all():
            if precis:
                if sup==0:
                    print("\nPoussins avec une fraction de l'orbite en dehors de 100km")
                print("poussin {} ; theta = +- {} pi ; distance max {} km".format(i, np.round(theta/np.pi,2), np.round(np.max(tab_N[i]/1e3),2)))
            sup += 1
        theta += 2*np.pi/(nb_poussin/2)

    theta = 0
    for i in range(nb_poussin): # totalement en dehors des 100km
        if (tab_N[i]>100e3).all():
            if precis:
                if sup2==0:
                    print("\nPoussins avec toute l'orbite en dehors de 100km")
                print("poussin {} ; theta = +- {} pi ; distance max {} km".format(i, np.round(theta/np.pi,2), np.round(np.max(tab_N[i]/1e3),2)))
            sup2+=1
        theta += 2*np.pi/(nb_poussin/2)

    print("\nPoussins en dehors des 100km : \n - sur une fraction de leur orbite :",sup)
    print(" - sur toute leur orbite :",sup2)
    print("\nPoussins dans les 100km de la poule :", nb_poussin - (sup+sup2))


#_______________________________________________________________
#__________________________ Execution __________________________
#_______________________________________________________________

dt = period/precision # tmp entre chaque pas de temps (en sec)

if X:
    N = int(X*period/dt) # orbite sur X periode
else:
    N = int(365*24*3600/dt) # duree totale de la mission

mother, nanos = Trajectories(_algo_verlet2) # calcul des trajectoires

#_______________________________________________________________
#____________________________ Plots ____________________________
#_______________________________________________________________

Outliers(precis=False)

if plot2D:
    Plot2D(mother, nanos)
if plot3D:
    Plot3D(mother, nanos)

if dist_std:
    if X:
        Dist_std(mother, nanos, individual=True, std=True)
    else:
        Dist_std(mother, nanos, individual=True, std=True, interval=(0,1.5))
        Dist_std(mother, nanos, individual=True, std=True, interval=(N*dt/period-1.5,N*dt/period))


if traj_ref_lune:
    Traj_animated(mother, nanos)
if animated:
    Relat_pos_animated(mother, nanos) # ref = 'lune' ou 'poule'  ('poule' par defaut)
