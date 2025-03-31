from modello_base import ModelloBase
import pandas as pd
# Passo 1. Analisi generale del dataframe - Valori univoci
# Passo 2. Ripulire il dataframe dalle variabili che non servono per l'analisi
# Passo 3. Ripulire il dataframe dalle variabili che contengono più valori nulli
# Passo 4. Per le variabili che contengono pochi valori nan,
# si effettuano delle sostituzioni:
# se i valori sono numerici si sceglie tra la media (con pochi valori outleirs)
# oppure la mediana. (Per pochi valori outliers s'intende minore del 15%/10%)
# se i valori degli outliers cresce notevolmente con la sostituzione:
# si può eseguire la mediana su gruppi (significativi) per migliorare la situazione
# Passo 5. È consigliato sostituire i valori stringhe con valori numerici (rimappatura
# Passo 6.	Possiamo modificare i nomi delle colonne (Facoltativa)


class ModelloTitanic(ModelloBase):

    def __init__(self, dataset_path):
        self.dataframe = pd.read_csv(dataset_path)
        self.dataframe_sistemato = self.sistemazione_dataframe()

    # Metodo di istanza per sistemazione del dataframe
    def sistemazione_dataframe(self):
        # Drop colonne inutili ai fini del modello
        variabili_da_droppare = ["name", "ticket", "fare", "cabin", "embarked", "home.dest", "boat", "body"]
        df_sistemato = self.dataframe.drop(variabili_da_droppare, axis=1).copy()
        # Drop dell'osservazione con tutti i valori nulli nan
        df_sistemato = df_sistemato.drop(index=1309, axis=0).copy()
        # Sostituzione valori nan di age con mediana
        #df_sistemato["age"] = df_sistemato["age"].fillna(df_sistemato["age"].median())
        df_sistemato["age"] = (df_sistemato.groupby(["pclass", "sex"])["age"]
                               .apply(lambda x: x.fillna(x.median()))).reset_index(level=[0,1], drop=True)
        # Rimapapatura colonna sex (0:female, 1:male)
        df_sistemato["sex"] = df_sistemato["sex"].map({"female":0, "male":1})
        # Modifica nomi delle colonne
        df_sistemato = df_sistemato.rename(columns={
            "pclass":"Classe Passeggero",
            "survived": "Sopravvissuto",
            "sex" : "Genere",
            "age" : "Età",
            "sibsp": "Fratelli/Coniugi",
            "parch": "Genitori/Figli"
        })

        # Conversione di tipo float in int
        for col in df_sistemato:
            df_sistemato[col] = df_sistemato[col].astype(int)

        return df_sistemato




# Utilizzo modello
modello = ModelloTitanic("../dataset/data_04.csv")
#modello.analisi_generali(modello.dataframe_sistemato)
#modello.analisi_valori_univoci(modello.dataframe_sistemato, ["Età", "Fratelli/Coniugi", "Genitori/Figli"])
#modello.individuazione_outliers(modello.dataframe_sistemato, ["Genere"])
#modello.dataframe_sistemato.to_csv("../dataset_sistemati/data_04.csv", index=False)
modello.analisi_indici_statistici(modello.dataframe)
