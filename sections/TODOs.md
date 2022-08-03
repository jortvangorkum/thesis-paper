TODOs:

Introduction:
- Voorbeeld toevoegen van hoe de incrementel computatie werkt
- Voorbeeld toevoegen specifiek van een boom
- Expliciter erin zetten dat de lookup gebasseerd op een lengte van de input, dat kunnen sneller doen door hashes te gebruiken
- De uitleg van de Zipper weglaten, alleen revereren

Bladzijde 20:
- Laat een voorbeeld van performance zien wat het gevolg is van al dit werk
  - Voorbeeld: cataSum zonder map en cataSum met map (wat het verschil in tijd is)

Cache addition policies:
- Voorbeeld uitwerken van een cache addition policy

Cache replacement policies:
- Voorbeeld meer uitwerken

Related work:
- Plaatsing: voor future work
- MemoTrie
- Jeroen Bransen - PhD thesis
- Umut Acar (en vele anderen)

Future work:
- Prioriteit geven aan future work (hoofdzaak/bijzaak en complexiteit/moeilijk)
- Hoe gaan we om met een nieuwe boom (geen updates), en kijken hoe de updates daarvan bepalen. Kan helpen bij compilers.

Pattern functors:
- datatypes die de pattern functors beschrijven = primitive type constructors

Sums of products:
- de implementatie gebruikt type level list, om dit te gebruiken heb je meer complexe type level. 
- de generic representatie is complexer door de list of list naar een kind conversie