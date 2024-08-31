import pandas as pd
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import re

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_training_data(df):
    training_data = []
    for _, row in df.iterrows():
        text = row['titulo']
        entities = []

        def find_entity(entity_type, pattern, existing_entities):
            match = re.finditer(pattern, text, re.IGNORECASE)
            for m in match:
                start, end = m.start(), m.end()
                if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
                    return (start, end, entity_type)
            return None
        
        model_entity = find_entity('MODEL', re.escape(row['modelo']), entities)
        if model_entity:
            entities.append(model_entity)
        
        entity_patterns = {
            'CPU': r'\b(Intel Core [^\s]+|AMD Ryzen [^\s]+)\b',
            'GPU': r'\b(GTX \d+\s*(?:Ti|Super)?|RTX \d+\s*(?:Ti|Super)?|Radeon\s*\w+\s*\d*\w*|NVIDIA\s*\w+\s*\d*\w*|Intel\s*(?:Iris\s*Xe|UHD|HD\s*Graphics)|\bAMD\s*\w+\s*\d*\w*|GT\d+\s*|MX\d+\s*)\b',
            'RAM': r'\b(\d+\s*GB RAM|\d+\s*G RAM|\d+\s*GB)\b',
            'SSD': r'\b(\d+\s*GB SSD|\d+\s*TB SSD)\b'
        }
        
        for entity_type, pattern in entity_patterns.items():
            entity = find_entity(entity_type, pattern, entities)
            if entity:
                entities.append(entity)
        
        if entities: 
            training_data.append((text, {"entities": entities}))
    
    return training_data

def train_model(training_data, iterations=100):
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
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/base-de-dados/notebooks_com_modelo_e_gamer.csv"
    output_dir = "modelo_ner_notebooks"
    
    df = load_data(csv_file)
    training_data = prepare_training_data(df)
    
    nlp = train_model(training_data)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
