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
                # Verifica se já existe uma entidade sobreposta
                if not any(e_start < end and start < e_end for e_start, e_end, _ in existing_entities):
                    return (start, end, entity_type)
            return None

        # Priorizar valores do CSV (modelo, CPU, GPU, RAM, SSD)
        entity_types = {
            'MODEL': row['modelo'],
            'CPU': row['CPU'],
            'GPU': row['GPU'],
            'RAM': row['RAM'],
            'SSD': row['SSD']
        }

        # Primeiro tenta encontrar os valores diretamente do CSV no texto
        for entity_type, value in entity_types.items():
            if pd.notna(value):  # Verifica se o valor não é nulo
                entity = find_entity(entity_type, re.escape(str(value)), entities)
                if entity:
                    entities.append(entity)

        # Se algum valor não foi encontrado, tenta usar expressões regulares
        entity_patterns = {
            'CPU': r'\b(Intel Core [^\s]+|AMD Ryzen [^\s]+)\b',
            'GPU': r'\b(GTX \d+\s*(?:Ti|Super)?|RTX \d+\s*(?:Ti|Super)?|Radeon\s*\w+\s*\d*\w*|NVIDIA\s*\w+\s*\d*\w*|Intel\s*(?:Iris\s*Xe|UHD|HD\s*Graphics)|\bAMD\s*\w+\s*\d*\w*|GT\d+\s*|MX\d+\s*)\b',
            'RAM': r'\b(\d+\s*GB RAM|\d+\s*G RAM|\d+\s*GB)\b',
            'SSD': r'\b(\d+\s*GB SSD|\d+\s*TB SSD)\b'
        }

        # Apenas tenta as expressões regulares se o valor do CSV não foi encontrado
        for entity_type, pattern in entity_patterns.items():
            if not any(e_type == entity_type for _, _, e_type in entities):  # Se a entidade não foi encontrada
                entity = find_entity(entity_type, pattern, entities)
                if entity:
                    entities.append(entity)

        if entities:
            training_data.append((text, {"entities": entities}))

    return training_data

def train_model(training_data, iterations=350):
    nlp = spacy.blank("pt")  # Carrega um modelo vazio para o idioma português
    ner = nlp.add_pipe("ner", last=True)
    
    # Adiciona as etiquetas de entidades ao modelo
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Treinamento
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # Desativa outros pipes durante o treinamento
        optimizer = nlp.begin_training()
        
        # Ajusta o tamanho do lote com um crescimento mais agressivo
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            
            # Tenta diferentes tamanhos de lote e crescimento
            batches = minibatch(training_data, size=compounding(8.0, 64.0, 1.01))
            
            for batch in batches:
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    
                    # Atualiza o modelo
                    nlp.update(
                        [example],  # Exemplo de treinamento
                        drop=0.3,   # Dropout mais baixo para permitir mais aprendizado
                        sgd=optimizer,
                        losses=losses
                    )
            print(f"Iteration {itn + 1}, Losses: {losses}")
    
    return nlp

def save_model(nlp, output_dir):
    nlp.to_disk(output_dir)

def main():
    csv_file = "/media/paulo-jaka/Extras/Machine-learning/notebook_sample_training_data.csv"
    output_dir = "modelo_ner_notebooks4_last"
    
    df = load_data(csv_file)
    training_data = prepare_training_data(df)
    
    nlp = train_model(training_data)
    save_model(nlp, output_dir)

if __name__ == "__main__":
    main()
