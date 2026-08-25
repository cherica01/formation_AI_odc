import math

entropie = 0 
proportion = [0.5,0.5]

for p in proportion :
    if p!= 0 :
        entropie = entropie - p * math.log2(p)
print("entropie=",entropie)