#!/usr/bin/env python3
"""
Script para analizar y verificar el balanceo de datos en el dataset de tenis.
Este script permite identificar y corregir problemas de balanceo de clases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import random
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def load_data(file_path):
    """Carga y verifica el archivo de datos."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
        data = pd.read_csv(file_path)
        logging.info(f"Datos cargados correctamente: {len(data)} registros")
        return data
    except Exception as e:
        logging.error(f"Error cargando datos: {e}")
        return None

def analyze_classes(data):
    """Analiza la distribución de clases en el dataset."""
    if 'winner' not in data.columns:
        logging.error("El dataset no contiene la columna 'winner'")
        return False
    
    class_counts = data['winner'].value_counts()
    total = len(data)
    
    print("\n=== DISTRIBUCIÓN DE CLASES ===")
    for class_val, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"Clase {class_val}: {count} registros ({percentage:.2f}%)")
    
    # Verificar si hay desbalance
    if len(class_counts) < 2:
        print("\n⚠️ PROBLEMA CRÍTICO: Solo hay una clase en los datos")
        print(f"Todos los registros tienen winner={class_counts.index[0]}")
        return False
    
    min_class = class_counts.min()
    max_class = class_counts.max()
    ratio = min_class / max_class
    
    if ratio < 0.8:
        print(f"\n⚠️ Se detectó desbalance de clases (ratio: {ratio:.2f})")
        if ratio < 0.2:
            print("Desbalance severo detectado. Se recomienda balancear los datos.")
    else:
        print(f"\n✅ Distribución de clases balanceada (ratio: {ratio:.2f})")
    
    return True

def balance_data(data, output_path=None):
    """Balancea los datos y opcionalmente guarda el resultado."""
    if 'winner' not in data.columns:
        logging.error("El dataset no contiene la columna 'winner'")
        return None
    
    class_counts = data['winner'].value_counts()
    
    # Si solo hay una clase
    if len(class_counts) == 1:
        only_class = class_counts.index[0]
        print(f"Solo existe la clase {only_class}. Balanceando mediante inversión...")
        
        # Copiar datos originales
        balanced_data = data.copy()
        
        # Crear conjunto invertido
        inverted_data = data.copy()
        
        # Intercambiar jugadores
        inverted_data['player1'], inverted_data['player2'] = inverted_data['player2'], inverted_data['player1']
        
        # Intercambiar estadísticas correspondientes
        if 'ranking_1' in inverted_data.columns and 'ranking_2' in inverted_data.columns:
            inverted_data['ranking_1'], inverted_data['ranking_2'] = inverted_data['ranking_2'], inverted_data['ranking_1']
        
        if 'winrate_1' in inverted_data.columns and 'winrate_2' in inverted_data.columns:
            inverted_data['winrate_1'], inverted_data['winrate_2'] = inverted_data['winrate_2'], inverted_data['winrate_1']
        
        # Cambiar el valor de winner al opuesto
        inverted_data['winner'] = 1 if only_class == 0 else 0
        
        # Combinar datos originales e invertidos
        balanced_data = pd.concat([balanced_data, inverted_data], ignore_index=True)
        
        # Mezclar aleatoriamente
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Datos balanceados: {len(balanced_data)} registros")
        print(f"Nueva distribución: {balanced_data['winner'].value_counts().to_dict()}")
        
        # Guardar si se especificó una ruta
        if output_path:
            balanced_data.to_csv(output_path, index=False)
            print(f"Datos balanceados guardados en: {output_path}")
        
        return balanced_data
    
    # Si hay desbalance entre las clases existentes
    min_class = class_counts.idxmin()
    maj_class = class_counts.idxmax()
    
    if class_counts[min_class] / class_counts[maj_class] < 0.8:
        print(f"Balanceando datos (upsampling de clase {min_class})...")
        
        # Separar clases
        min_class_data = data[data['winner'] == min_class]
        maj_class_data = data[data['winner'] == maj_class]
        
        # Calcular cuántos registros añadir
        n_samples = len(maj_class_data) - len(min_class_data)
        
        # Upsample (sobremuestrear) la clase minoritaria
        upsampled = min_class_data.sample(n=n_samples, replace=True, random_state=42)
        
        # Combinar con los datos originales
        balanced_data = pd.concat([data, upsampled], ignore_index=True)
        
        # Mezclar aleatoriamente
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Datos balanceados: {len(balanced_data)} registros")
        print(f"Nueva distribución: {balanced_data['winner'].value_counts().to_dict()}")
        
        # Guardar si se especificó una ruta
        if output_path:
            balanced_data.to_csv(output_path, index=False)
            print(f"Datos balanceados guardados en: {output_path}")
        
        return balanced_data
    
    print("Los datos ya están suficientemente balanceados. No se requieren cambios.")
    return data

def randomize_winners(data, output_path=None):
    """
    Aleatoriza la asignación winner/loser en datos con formato de Jeff Sackmann.
    Este enfoque es útil cuando el dataset original tiene winner_name/loser_name.
    """
    if 'winner_name' not in data.columns or 'loser_name' not in data.columns:
        print("Este dataset no tiene formato Jeff Sackmann (winner_name/loser_name).")
        return data
    
    print("Detectado formato Jeff Sackmann. Aleatorizando asignación player1/player2...")
    
    # Crear nuevo DataFrame
    randomized = pd.DataFrame()
    
    # Para cada fila, aleatorizar la asignación
    for i, row in data.iterrows():
        # Aleatorizar (aproximadamente 50/50)
        if random.random() > 0.5:
            # winner_name es player1
            new_row = {
                'player1': row['winner_name'],
                'player2': row['loser_name'],
                'winner': 0
            }
            
            # Mapear rankings si existen
            if 'winner_rank' in row and 'loser_rank' in row:
                new_row['ranking_1'] = row['winner_rank']
                new_row['ranking_2'] = row['loser_rank']
        else:
            # loser_name es player1
            new_row = {
                'player1': row['loser_name'],
                'player2': row['winner_name'],
                'winner': 1
            }
            
            # Mapear rankings si existen
            if 'winner_rank' in row and 'loser_rank' in row:
                new_row['ranking_1'] = row['loser_rank']
                new_row['ranking_2'] = row['winner_rank']
        
        # Copiar otros campos relevantes
        for field in ['surface', 'tourney_name', 'tourney_date']:
            if field in row:
                new_field = field
                if field == 'tourney_name':
                    new_field = 'tournament'
                elif field == 'tourney_date':
                    new_field = 'match_date'
                
                new_row[new_field] = row[field]
        
        # Añadir fila al nuevo DataFrame
        randomized = pd.concat([randomized, pd.DataFrame([new_row])], ignore_index=True)
    
    # Calcular winrates basados en los datos randomizados
    player_stats = {}
    all_players = set(randomized['player1'].tolist() + randomized['player2'].tolist())
    
    for player in all_players:
        # Victorias como player1
        p1_matches = randomized[randomized['player1'] == player]
        p1_wins = p1_matches[p1_matches['winner'] == 0].shape[0]
        
        # Victorias como player2
        p2_matches = randomized[randomized['player2'] == player]
        p2_wins = p2_matches[p2_matches['winner'] == 1].shape[0]
        
        # Total de partidos y victorias
        total_matches = len(p1_matches) + len(p2_matches)
        total_wins = p1_wins + p2_wins
        
        if total_matches > 0:
            winrate = (total_wins / total_matches) * 100
            player_stats[player] = winrate
    
    # Aplicar winrates
    randomized['winrate_1'] = randomized['player1'].map(player_stats)
    randomized['winrate_2'] = randomized['player2'].map(player_stats)
    
    print(f"Datos aleatorizados: {len(randomized)} registros")
    print(f"Distribución: {randomized['winner'].value_counts().to_dict()}")
    
    if output_path:
        randomized.to_csv(output_path, index=False)
        print(f"Datos aleatorizados guardados en: {output_path}")
    
    return randomized

def visualize_data(data):
    """Genera visualizaciones para analizar el dataset."""
    if data is None or data.empty:
        logging.error("No hay datos para visualizar")
        return
    
    print("\nGenerando visualizaciones...")
    
    # Crear directorio para visualizaciones
    vis_dir = "tennis_data_analysis"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Distribución de clases
    plt.figure(figsize=(10, 6))
    sns.countplot(x='winner', data=data)
    plt.title('Distribución de Variable Objetivo (winner)')
    plt.xlabel('Ganador (0=player1, 1=player2)')
    plt.ylabel('Cantidad de Partidos')
    plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Distribución por superficie
    if 'surface' in data.columns:
        plt.figure(figsize=(12, 6))
        surface_counts = data['surface'].value_counts()
        sns.barplot(x=surface_counts.index, y=surface_counts.values)
        plt.title('Distribución de Partidos por Superficie')
        plt.xlabel('Superficie')
        plt.ylabel('Cantidad de Partidos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'surface_distribution.png'))
        plt.close()
        
        # 2.1 Tasas de victoria por superficie
        plt.figure(figsize=(12, 6))
        surface_win_rate = data.groupby('surface')['winner'].mean() * 100
        sns.barplot(x=surface_win_rate.index, y=surface_win_rate.values)
        plt.title('Tasa de Victoria de player2 por Superficie')
        plt.xlabel('Superficie')
        plt.ylabel('% Victoria de player2 (winner=1)')
        plt.xticks(rotation=45)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'surface_win_rate.png'))
        plt.close()
    
    # 3. Rankings y resultados
    if 'ranking_1' in data.columns and 'ranking_2' in data.columns:
        # 3.1 Crear variable ranking_diff
        data['ranking_diff'] = data['ranking_1'] - data['ranking_2']
        
        # 3.2 Visualizar ranking_diff vs. resultados
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x='ranking_diff', hue='winner', bins=30, element="step", 
                    common_norm=False, alpha=0.5)
        plt.title('Diferencia de Ranking vs. Resultado')
        plt.xlabel('Diferencia de Ranking (player1 - player2)')
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(vis_dir, 'ranking_diff_results.png'))
        plt.close()
        
        # 3.3 Crear variable para mejor ranking
        data['better_ranked_wins'] = ((data['ranking_1'] < data['ranking_2']) & (data['winner'] == 0)) | \
                                    ((data['ranking_1'] > data['ranking_2']) & (data['winner'] == 1))
                                    
        # 3.4 Calcular porcentaje de victorias del mejor rankeado
        better_ranked_win_pct = data['better_ranked_wins'].mean() * 100
        print(f"\nPorcentaje de victorias del mejor rankeado: {better_ranked_win_pct:.2f}%")
        
        # Visualizar en gráfico de pastel
        plt.figure(figsize=(8, 8))
        plt.pie([better_ranked_win_pct, 100 - better_ranked_win_pct], labels=['Gana mejor rankeado', 'Gana peor rankeado'],
                autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], explode=[0.1, 0])
        plt.title('Victorias según Ranking')
        plt.savefig(os.path.join(vis_dir, 'better_ranked_wins.png'))
        plt.close()
    
    # 4. Tasas de victoria y resultados
    if 'winrate_1' in data.columns and 'winrate_2' in data.columns:
        # 4.1 Crear variable winrate_diff
        data['winrate_diff'] = data['winrate_1'] - data['winrate_2']
        
        # 4.2 Visualizar winrate_diff vs. resultados
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x='winrate_diff', hue='winner', bins=30, element="step", 
                    common_norm=False, alpha=0.5)
        plt.title('Diferencia de Tasa de Victoria vs. Resultado')
        plt.xlabel('Diferencia de Winrate (player1 - player2)')
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(vis_dir, 'winrate_diff_results.png'))
        plt.close()
        
        # 4.3 Crear variable para mejor tasa de victoria
        data['better_winrate_wins'] = ((data['winrate_1'] > data['winrate_2']) & (data['winner'] == 0)) | \
                                     ((data['winrate_1'] < data['winrate_2']) & (data['winner'] == 1))
                                    
        # 4.4 Calcular porcentaje de victorias del jugador con mejor winrate
        better_winrate_win_pct = data['better_winrate_wins'].mean() * 100
        print(f"Porcentaje de victorias del jugador con mejor winrate: {better_winrate_win_pct:.2f}%")
        
        # Visualizar en gráfico de pastel
        plt.figure(figsize=(8, 8))
        plt.pie([better_winrate_win_pct, 100 - better_winrate_win_pct], 
                labels=['Gana mejor winrate', 'Gana peor winrate'],
                autopct='%1.1f%%', colors=['#2196F3', '#FF9800'], explode=[0.1, 0])
        plt.title('Victorias según Tasa de Victoria')
        plt.savefig(os.path.join(vis_dir, 'better_winrate_wins.png'))
        plt.close()
    
    # 5. Heatmap de correlaciones
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation = data[numeric_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'correlation_matrix.png'))
        plt.close()
    
    print(f"Visualizaciones guardadas en el directorio: {os.path.abspath(vis_dir)}")

def main():
    parser = argparse.ArgumentParser(description="Analizador de datos para predicción de tenis")
    parser.add_argument('--file', type=str, default='tennis_matches.csv', help='Ruta al archivo de datos')
    parser.add_argument('--balance', action='store_true', help='Balancear clases en los datos')
    parser.add_argument('--randomize', action='store_true', help='Aleatorizar asignación winner/loser (formato Jeff Sackmann)')
    parser.add_argument('--visualize', action='store_true', help='Generar visualizaciones')
    parser.add_argument('--output', type=str, help='Ruta para guardar datos procesados')
    
    args = parser.parse_args()
    
    # Generar nombre de archivo de salida si no se especificó
    if args.balance or args.randomize:
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.balance:
                args.output = f"tennis_matches_balanced_{timestamp}.csv"
            elif args.randomize:
                args.output = f"tennis_matches_randomized_{timestamp}.csv"
    
    # Cargar datos
    data = load_data(args.file)
    if data is None:
        return
    
    # Mostrar información básica
    print(f"\nDimensiones del dataset: {data.shape}")
    print(f"Columnas: {', '.join(data.columns)}")
    
    # Analizar distribución de clases
    if 'winner' in data.columns:
        analyze_classes(data)
    elif 'winner_name' in data.columns and 'loser_name' in data.columns:
        print("\nDetectado formato Jeff Sackmann (winner_name/loser_name)")
        print("Se recomienda usar --randomize para convertir a formato estándar")
    
    # Realizar operaciones según parámetros
    processed_data = data
    
    if args.randomize:
        processed_data = randomize_winners(data, args.output)
    elif args.balance and 'winner' in data.columns:
        processed_data = balance_data(data, args.output)
    
    # Generar visualizaciones
    if args.visualize:
        visualize_data(processed_data)

if __name__ == "__main__":
    main()