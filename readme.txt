##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
				Projet NOIRE 2020
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Ce fichier est lié au programme en Python main.py qui permet de larguer et propager une constellation de satellites autour de la Lune.

Une série de paramètres peuvent être modifiés au sein même du programme, pour obtenir les résultats et graphiques demandés (lignes 14 à 26). Les constantes physiques utilisées se trouvent au début du code (lignes 34 à 40).


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La définition principale du programme est "Trajectories", elle permet de larguer et propager les satellites grâce à un algorithme d'intégration défini. 
Pour le moment, les propagations fonctionnent avec tous les algorithmes mais seul "verlet2" est adapté à la propagation ET aux largages. De plus c'est celui qui permet de réduire au maximum les erreurs numériques, il est donc fortement conseillé de n'utiliser que celui-ci, les autres nécessitant des modifications au niveau de l'ajout des impulsions pour un bon fonctionnement.


Quelques définitions viennent ensuite pour visualiser les résultats : 
 - "Plot2D" et "Plot3D" pour afficher les résultats dans la dimension voulue
 - "Dist_std" pour afficher les positions relatives des poussins par rapport à la mère. 

Deux définitions permettent de visualiser ces trajectoires d'une manière interactive : 
 - "Traj_animated" qui permet de visualiser les lignes des trajectoires des poussins et de la mère dans le référentiel lunaire 
 - "Relat_pos_animated" qui permet de visualiser les positions des poussins à tout instants dans un référentiel choisi.

La définition "Outliers" permet d'identifier les poussins qui sont sortis de la limite des 100km et de savoir s'ils en sont sortis pendant une partie de leur période ou s'ils sont définitivement en dehors des 100km de la mère.


PS : pour obtenir des informations sur le fonctionnement des définitions, une simple commande help(definition_name) dans le terminal python peut être effectuée. Des annotations sont également présentes tout au long des codes pour suivre le fil des actions.


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Les graphiques et films des animations ont été mis dans le dossier Results au cas où le programme serait dans l'impossibilité d'être lancé ou simplement pour visualiser les résultats sans lancer le programme.


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Une explication compréhensible pour une implémentation rapide des algorithmes de verlet et euler peut se trouver sur le site suivant :
https://femto-physique.fr/omp/methode-de-verlet.php


