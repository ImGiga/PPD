import numpy as np
import matplotlib.pyplot as plt

cZiel =  np.array([1,25,50,100,150,200,250]) 
cStamm = 250            # mu g / mu l
vZiel = 250             # mu l

vStammExakt = vZiel * cZiel / cStamm                # Gedanke: vStamm * cStamm / (vZiel = vGesamt) = cZiel

vStamm = (np.round(vStammExakt*10) + (np.round(vStammExakt*10) % 2)) / 10   # Wegen Genauigkeit von 0,2 mu l
vZielNeu = vStamm * cStamm / cZiel                                          # vZiel als variable Größe
vChloBenz = vZielNeu - vStamm                                                  

cTest = vStamm * cStamm / (vStamm + vChloBenz)

for j, cZ in enumerate(cZiel):
    print(f"Ziel-Konzentration: {cZ}\tStamm-Volumen: {vStamm[j]}\tChlorBenzol-Volumen: {vChloBenz[j]}\tResultierende Konzentration: {cTest[j]}\n")

print(f"Gesamtvolumen der Stammlösung: {sum(vStamm)-vStamm[0]}\tvon ChlorBenzol: {sum(vChloBenz)}")