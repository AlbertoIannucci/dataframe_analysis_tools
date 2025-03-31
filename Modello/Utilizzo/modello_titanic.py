from modello_base import ModelloBase
import pandas as pd

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
modello.analisi_generali(modello.dataframe_sistemato)
modello.analisi_valori_univoci(modello.dataframe_sistemato, ["Età", "Fratelli/Coniugi", "Genitori/Figli"])
modello.individuazione_outliers(modello.dataframe_sistemato, ["Genere"])
modello.analisi_indici_statistici(modello.dataframe)
