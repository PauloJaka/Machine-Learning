import pandas as pd
import re

# Array com os nomes das marcas
brands: list = [
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

def extract_model_based_on_brand(title):
    if not isinstance(title, str):
        return None
    
    # Primeiramente, tentamos capturar a geração do iPad
    ipad_generation = extract_ipad_generation(title)
    if ipad_generation:
        return ipad_generation
    
    # Caso contrário, procurar o modelo baseado na marca
    for brand in brands:
        match = re.search(fr'\b{brand}\b', title, re.IGNORECASE)
        if match:
            # Pega apenas as palavras após a marca, sem incluir a própria marca
            after_brand = title[match.end():].strip()
            model = ' '.join(after_brand.split()[:3])  # Pega as 2-3 primeiras palavras após a marca
            return model  # Retorna apenas o modelo, sem a marca
    return None

# Função para extrair a geração do iPad
def extract_ipad_generation(title):
    if not isinstance(title, str):
        return None
    
    match = re.search(r'iPad\s?(Pro)?\s?(\d+ª|\d+th)\s?(geração|generation)?', title, re.IGNORECASE)
    if match:
        generation = match.group(0)
        return generation
    return None

def extract_ram(title):
    if not isinstance(title, str):
        return None
    
    match = re.search(r'(\d+)\s*(GB|gb|Gb|gB)\s*(RAM|ram)?', title, re.IGNORECASE)
    if match and 4 <= int(match.group(1)) <= 64:  
        return match.group(1) + " GB"
    return None

def extract_storage_capacity(title):
    if not isinstance(title, str):
        return None

    # Regex aprimorada para capturar armazenamento
    match = re.search(r'(\d+)\s*(GB|TB|tb|Gb|Tb)', title, re.IGNORECASE)
    if match and int(match.group(1)) >= 32:  # Verifica se o valor é válido para armazenamento
        return match.group(1) + " " + match.group(2).upper()
    return None

# Carregar CSV
csv_path = '/home/paulo-jaka/Downloads/datasets_trabalho/tablets_new_training_data.csv'  # Substitua pelo caminho correto do seu arquivo CSV
df = pd.read_csv(csv_path)

# Aplicar as funções de extração
df['model'] = df['title'].apply(extract_model_based_on_brand)
df['RAM'] = df['title'].apply(extract_ram)
df['storage_capacity'] = df['title'].apply(extract_storage_capacity)

# Salvar o novo CSV com as colunas extraídas
output_path = '/media/paulo-jaka/Extras/Machine-learning/base-de-dados/dados_tablet.csv'
df.to_csv(output_path, index=False)

print("Informações extraídas e salvas em", output_path)
