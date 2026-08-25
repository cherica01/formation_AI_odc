import math

entropie = 0 
proportion = [0.5,0.5]

for p in proportion :
    if p!= 0 :
        entropie = entropie - p * math.log2(p)
print("entropie=",entropie)
entropie_parent = 1.0

gain = entropie_parent - (
    (4 / 10) * 0.811 +
    (2 / 10) * 0 +
    (4 / 10) * 1
)

print("Gain d'information =", gain)


for z in [-2, 0, 2]:
    resultat = 1 / (1 + math.exp(-z))
    print("sigmoide de z =", z, "=>", resultat)