import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import re

KNOWN_BRANDS = [
    "ACER", "ASUS", "SAMSUNG", "Dell", "Positivo", "Lenovo", "VAIO",
    "HP", "Apple", "Multilaser", "Anvazise", "ASHATA", "Santino", "MSI",
    "Marca Fácil", "Microsoft", "AWOW", "Gateway", "Compaq", "DAUERHAFT",
    "SGIN", "Luqeeg", "Kiboule", "LG", "Panasonic", "Focket", "Toughbook",
    "LTI", "GIGABYTE", "Octoo", "Chip7 Informática", "GLOGLOW", "GOLDENTEC",
    "KUU", "HEEPDD", "Adamantiun", "Naroote", "Jectse", "Heayzoki", "Galaxy",
    "Motorola", "Xiaomi", "Nokia", "Poco", "realme", "Infinix", "Blu",
    "Gshield", "Geonav", "Redmi", "Gorila Shield", "intelbras", "TCL",
    "Tecno", "Vbestlife", "MaiJin", "SZAMBIT", "Otterbox", "Sony",
    "HAIZ", "HUAWEI", "HAYLOU", "Amazfit", "Ticwatch", "Legado Engenharia",
    "Microwear", "Lefal Cold", "MPOWER", "Kaymcixs", "Garmin", "123Smart",
    "Technos", "IWO", "Polar", "Mormaii", "xsmart",
    "EIGIIS", "Beyamis", "Hrich", "ANCOOL", "Dpofirs", "C7 company",
    "ShieldForce", "FIT IT", "Blackview", "KALINCO", "Danet", "LDFAS",
    "VINGVO", "MIJOBS", "KADES", "Naroote", "Gusfeliz",
    "Fossil", "Withings", "Suunto", "Mobvoi", "Amazfit", "Garmin", "Fitbit",
    "Huawei", "Apple", "Samsung", "TicWatch", "Polar", "Tag Heuer", "Casio",
    "Amazfit", "Garmin", "Suunto", "Huawei", "Fitbit", "Amazfit", "Withings",
    "Fossil", "Huawei", "Michael Kors", "Misfit", "TomTom", "Jawbone", "Zepp","Philco",
    "Britânia", "Roku", "Dolby", "Philips", "Semp"
]


def load_data(file_path):
    return pd.read_csv(file_path)

def find_entity(entity_type, pattern, text, existing_entities):
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    for m in matches:
        start, end = m.start(), m.end()
        if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
            return (start, end, entity_type)
    return None

def extract_entities(text, row):
    entities = []
    
    for brand in KNOWN_BRANDS:
        brand_entity = find_entity('BRAND', re.escape(brand), text, entities)
        if brand_entity:
            entities.append(brand_entity)
            break
    
    # Extract model
    model_entity = find_entity('MODEL', re.escape(row['modelo']), text, entities)
    if model_entity:
        entities.append(model_entity)
    
    return entities

def prepare_training_data(df):
    df['title'] = df['title'].astype(str).fillna('')
    df['brand'] = df['brand'].astype(str).fillna('')
    df['modelo'] = df['modelo'].astype(str).fillna('')
    training_data = []
    for _, row in df.iterrows():
        text = row['title']
        entities = extract_entities(text, row)
        if entities:
            training_data.append((text, {"entities": entities}))
    
    return training_data

def train_model(training_data, iterations=150): 
    nlp = spacy.blank("pt")
    ner = nlp.add_pipe("ner", last=True)
    
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")
    
    return nlp

def save_model(nlp, output_dir):
    nlp.to_disk(output_dir)

def main():
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/base-de-dados/smarthwatchs_dados_treino2.csv"
    output_dir = "modelo_ner_smartwatches"
    
    df = load_data(csv_file)
    training_data = prepare_training_data(df)
    
    # Print out some of the training data for debugging
    for text, annotations in training_data[:5]:  # Print first 5 for quick check
        print(f"Training data sample: {text} - {annotations}")

    nlp = train_model(training_data)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
